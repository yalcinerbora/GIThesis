#include "ThesisSolution.h"
#include "Globals.h"
#include "SceneI.h"
#include "DrawBuffer.h"
#include "Macros.h"
#include "Camera.h"
#include "DeferredRenderer.h"
#include "SceneLights.h"
#include <cuda_gl_interop.h>

const size_t ThesisSolution::InitialObjectGridSize = 256;
const float ThesisSolution::CascadeSpan = 0.6f;
const uint32_t ThesisSolution::CascadeDim = 512;

const TwEnumVal ThesisSolution::renderSchemeVals[] = 
{ 
	{ GI_DEFERRED, "Deferred" }, 
	{ GI_LIGHT_INTENSITY, "LI Buffer Only"},
	{ GI_SVO_LEVELS, "Render SVO Levels"},
	{ GI_VOXEL_PAGE, "Render Voxel Page" },
	{ GI_VOXEL_CACHE2048, "Render Voxel Cache 2048" },
	{ GI_VOXEL_CACHE1024, "Render Voxel Cache 1024" },
	{ GI_VOXEL_CACHE512, "Render Voxel Cache 512" }
};

ThesisSolution::ThesisSolution(DeferredRenderer& dRenderer, const IEVector3& intialCamPos)
	: currentScene(nullptr)
	, dRenderer(dRenderer)
	, vertexDebugVoxel(ShaderType::VERTEX, "Shaders/VoxRender.vert")
	, vertexDebugWorldVoxel(ShaderType::VERTEX, "Shaders/VoxRenderWorld.vert")
	, fragmentDebugVoxel(ShaderType::FRAGMENT, "Shaders/VoxRender.frag")
	, vertexVoxelizeObject(ShaderType::VERTEX, "Shaders/VoxelizeGeom.vert")
	, geomVoxelizeObject(ShaderType::GEOMETRY, "Shaders/VoxelizeGeom.geom")
	, fragmentVoxelizeObject(ShaderType::FRAGMENT, "Shaders/VoxelizeGeom.frag")
	, computeVoxelizeCount(ShaderType::COMPUTE, "Shaders/VoxelizeGeomCount.glsl")
	, computePackObjectVoxels(ShaderType::COMPUTE, "Shaders/PackObjectVoxels.glsl")
	, computeDetermineVoxSpan(ShaderType::COMPUTE, "Shaders/DetermineVoxSpan.glsl")
	, bar(nullptr)
	, renderScheme(GI_VOXEL_PAGE)
	//, renderScheme(GI_VOXEL_CACHE2048)
	, gridInfoBuffer(1)
	, voxelNormPosBuffer(512)
	, voxelColorBuffer(512)
	, voxelOctree()
	, traceType(0)
{
	renderType = TwDefineEnum("RenderType", renderSchemeVals, GI_END);
	gridInfoBuffer.AddData({});
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		voxelScenes.emplace_back(intialCamPos, CascadeSpan * (0x1 << i), CascadeDim);
	}
}

ThesisSolution::~ThesisSolution()
{}

bool ThesisSolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}

void ThesisSolution::Init(SceneI& s)
{
	// Reset GICudaScene
	voxelCaches.clear();
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		voxelScenes[i].Reset();
		voxelCaches.emplace_back();
	}
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	currentScene = &s;

	// Voxelization
	// and Voxel Cache Creation
	double voxelTotaltime = 0.0;
	Array32<MeshBatchI*> batches = currentScene->getBatches();
	for(unsigned int i = 0; i < batches.length; i++)
	{
		for(unsigned int j = 0; j < GI_CASCADE_COUNT; j++)
		{
			uint32_t multiplier = 0x1 << j;
			voxelCaches[j].cache.emplace_back(InitialObjectGridSize, batches.arr[i]->VoxelCacheMax(j));
			voxelTotaltime += Voxelize(voxelCaches[j].cache.back(), 
									   batches.arr[i],
									   CascadeSpan * multiplier, 
									   multiplier, 
									   j == 0);
		}
	}

	for(unsigned int i = 0; i < voxelCaches.size(); i++)
	{
		uint32_t totalCount = 0;
		double totalSize = 0.0f;
		for(unsigned int j = 0; j < voxelCaches[i].cache.size(); j++)
		{
			totalCount += voxelCaches[i].cache[j].batchVoxCacheCount;
			totalSize += voxelCaches[i].cache[j].batchVoxCacheSize;
		}
		voxelCaches[i].totalCacheCount = totalCount;
		voxelCaches[i].totalCacheSize = totalSize;
	}
	GI_LOG("Scene voxelization completed. Elapsed time %f ms", voxelTotaltime);
	
	// Voxel Page System Linking
	for(unsigned int j = 0; j < GI_CASCADE_COUNT; j++)
		LinkCacheWithVoxScene(voxelScenes[j], voxelCaches[j], 1.0f);
	
	// Allocators Link
	// Ordering is reversed svo tree needs cascades from other to inner
	std::vector<GICudaAllocator*> allocators;
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
		allocators.push_back(voxelScenes[GI_CASCADE_COUNT - i - 1].Allocator());

	voxelOctree.LinkAllocators(Array32<GICudaAllocator*>{allocators.data(), GI_CASCADE_COUNT},
							   currentScene->SVOTotalSize(),
							   currentScene->SVOLevelSizes());
	svoRenderLevel = voxelOctree.SVOConsts().totalDepth;

	// Memory Usage Total
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		GI_LOG("Voxel Sytem #%d Total Memory Usage %f MB", i,
			   static_cast<double>(voxelScenes[i].AllocatorMemoryUsage()) / 1024.0 / 1024.0);
	}
	GI_LOG("Voxel Octree Sytem Total Memory Usage %f MB",
		   static_cast<double>(voxelOctree.MemoryUsage()) / 1024.0 / 1024.0);

	// Tw Bar Creation
	bar = TwNewBar("ThesisGI");
	TwDefine(" ThesisGI refresh=0.01 ");

	// FPS Show
	TwAddVarRO(bar, "fTime", TW_TYPE_DOUBLE, &frameTime,
			   " label='Frame(ms)' precision=2 help='Frame Time in milliseconds.' ");
	TwAddVarRW(bar, "rendering", renderType,
			   &renderScheme,
			   " label='Render' help='Change what to show on screen' ");
	TwAddVarRW(bar, "giOn", TW_TYPE_BOOLCPP,
			   &giOn,
			   " label='GI On' help='Cone Tracing GI On off' ");
	TwAddSeparator(bar, NULL, NULL);
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		std::string start("label = 'Cascade#");
		start += std::to_string(i);
		std::string endCount(" Count' group='Voxel Cache' help='Cache voxel count.' ");
		std::string endSize(" Size(MB)' group='Voxel Cache' precision=2 help='Cache voxel total size in megabytes.' ");
		TwAddVarRO(bar, (std::string("voxCache") + std::to_string(i)).c_str(), TW_TYPE_UINT32, &voxelCaches[i].totalCacheCount,
				   (start + endCount).c_str());
		TwAddVarRO(bar, (std::string("voxCacheSize") + std::to_string(i)).c_str(), TW_TYPE_DOUBLE, &voxelCaches[i].totalCacheSize,
				   (start + endSize).c_str());
	}
	TwAddSeparator(bar, NULL, NULL);
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		std::string start("label = 'Cascade#");
		start += std::to_string(i);
		std::string endCount(" Count' group='Voxel Octree' help='Voxel count in octree.' ");
		std::string endSize(" Size(MB)' group='Voxel Octree' precision=2 help='Octree Voxel total size in megabytes.' ");

		TwAddVarRO(bar, (std::string("voxUsed") + std::to_string(i)).c_str(), TW_TYPE_UINT32, &voxelCaches[i].voxOctreeCount,
				   (start + endCount).c_str());
		TwAddVarRO(bar, (std::string("voxUsedSize") + std::to_string(i)).c_str(), TW_TYPE_DOUBLE, &voxelCaches[i].voxOctreeSize,
				   (start + endSize).c_str());
	}
	TwAddSeparator(bar, NULL, NULL);
	TwAddVarRO(bar, "ioTime", TW_TYPE_DOUBLE, &ioTime,
			   " label='I-O Time (ms)' group='Timings' precision=2 help='Voxel Include Exclude Timing per frame.' ");
	TwAddVarRO(bar, "updateTime", TW_TYPE_DOUBLE, &transformTime,
			   " label='Update Time (ms)' group='Timings' precision=2 help='Voxel Grid Update Timing per frame.' ");
	TwAddVarRO(bar, "svoReconTime", TW_TYPE_DOUBLE, &svoTime,
			   " label='SVO Time (ms)' group='Timings' precision=2 help='SVO Reconstruct Timing per frame.' ");
	TwAddVarRO(bar, "transferTime", TW_TYPE_DOUBLE, &debugVoxTransferTime,
			   " label='Dbg Transfer Time (ms)' group='Timings' precision=2 help='Voxel Copy to OGL Timing.' ");
	TwDefine(" ThesisGI size='325 430' ");
	TwDefine(" ThesisGI valueswidth=fit ");
}

void ThesisSolution::Release()
{
	if(bar) TwDeleteBar(bar);
	bar = nullptr;
}

double ThesisSolution::Voxelize(VoxelObjectCache& cache,
								MeshBatchI* batch,
								float gridSpan, unsigned int minSpanMultiplier,
								bool isInnerCascade)
{
	cache.objectGridInfo.Resize(batch->DrawCount());

	// Timing Voxelization Process
	GLuint queryID;
	glGenQueries(1, &queryID);
	glBeginQuery(GL_TIME_ELAPSED, queryID);

	//
	glClear(GL_COLOR_BUFFER_BIT);
	DrawBuffer& dBuffer = batch->getDrawBuffer();
	VoxelRenderTexture voxelRenderTexture;

	// Determine Voxel Sizes
	computeDetermineVoxSpan.Bind();
	cache.objectGridInfo.Resize(batch->DrawCount());
	batch->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
	cache.objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	glUniform1ui(U_TOTAL_OBJ_COUNT, static_cast<GLuint>(batch->DrawCount()));
	glUniform1f(U_MIN_SPAN, batch->MinSpan());
	glUniform1ui(U_MAX_GRID_DIM, VOXEL_GRID_SIZE);

	size_t blockCount = (batch->DrawCount() / 128);
	size_t factor = ((batch->DrawCount() % 128) == 0) ? 0 : 1;
	blockCount += factor;
	glDispatchCompute(static_cast<GLuint>(blockCount), 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	cache.objectGridInfo.RecieveData(batch->DrawCount());

	// Buffers
	cameraTransform.Bind();
	dBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();
	cache.objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	
	// Render Objects to Voxel Grid
	// Use MSAA to prevent missing small triangles on voxels
	// (testing conservative rendering on maxwell)
	glEnable(GL_MULTISAMPLE);
	//glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);

	// State
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDepthMask(false);
	glStencilMask(0x0000);
	glColorMask(false, false, false, false);
	glViewport(0, 0, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE);

	// Reset Cache
	cache.voxelCacheUsageSize.CPUData()[0] = 0;
	cache.voxelCacheUsageSize.SendData();

	// isMip
	std::vector<GLuint> isMip(batch->DrawCount(), 0);
	for(unsigned int i = 0; i < isMip.size(); i++)
	{
		if(cache.objectGridInfo.CPUData()[i].span < batch->MinSpan() * minSpanMultiplier)
		{
			isMip[i] = (isInnerCascade) ? 0 : 1;
			cache.objectGridInfo.CPUData()[i].span = batch->MinSpan() * minSpanMultiplier;
		}
			
	}
	cache.objectGridInfo.SendData();

	// For Each Object
	voxelRenderTexture.Clear();
	for(unsigned int i = 0; i < batch->DrawCount(); i++)
	{
		// Skip objects that cant fit
		if(cache.objectGridInfo.CPUData()[i].span != batch->MinSpan() * minSpanMultiplier)
			continue;

		// First Call Voxelize over 3D Texture
		batch->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
		voxelRenderTexture.BindAsImage(I_VOX_WRITE, GL_WRITE_ONLY);
		vertexVoxelizeObject.Bind();
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		geomVoxelizeObject.Bind();
		fragmentVoxelizeObject.Bind();
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		batch->getGPUBuffer().Bind();

		// Material Buffer we need to fetch color from material
		dBuffer.BindMaterialForDraw(i);

		// Draw Call
		glDrawElementsIndirect(GL_TRIANGLES,
							   GL_UNSIGNED_INT,
							   (void *) (i * sizeof(DrawPointIndexed)));


		// Reflect Changes for the next process
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		// Second Call: Determine voxel count
		// We need to set viewport coords to match the voxel dims
		const AABBData& objAABB = batch->getDrawBuffer().getAABBBuffer().CPUData()[i];
		GLuint voxDimX, voxDimY, voxDimZ;
		voxDimX = static_cast<GLuint>(std::floor((objAABB.max.getX() - objAABB.min.getX()) / cache.objectGridInfo.CPUData()[i].span)) + 1;
		voxDimY = static_cast<GLuint>(std::floor((objAABB.max.getY() - objAABB.min.getY()) / cache.objectGridInfo.CPUData()[i].span)) + 1;
		voxDimZ = static_cast<GLuint>(std::floor((objAABB.max.getZ() - objAABB.min.getZ()) / cache.objectGridInfo.CPUData()[i].span)) + 1;

		computeVoxelizeCount.Bind();
		voxelRenderTexture.BindAsImage(I_VOX_READ, GL_READ_ONLY);
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		glUniform3ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ);
		glDispatchCompute(voxDimX + 7 / 8, voxDimY + 7 / 8, voxDimZ + 7 / 8);

		// Reflect Voxel Size
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

		// Create sparse voxel array according to the size of voxel count
		// Last Call: Pack Draw Calls to the buffer
		computePackObjectVoxels.Bind();
		cache.voxelNormPos.BindAsShaderStorageBuffer(LU_VOXEL_NORM_POS);
		cache.voxelIds.BindAsShaderStorageBuffer(LU_VOXEL_IDS);
		cache.voxelRenderData.BindAsShaderStorageBuffer(LU_VOXEL_RENDER);
		cache.voxelCacheUsageSize.BindAsShaderStorageBuffer(LU_INDEX_CHECK);
		voxelRenderTexture.BindAsImage(I_VOX_READ, GL_READ_WRITE);
		glUniform1ui(U_OBJ_TYPE, static_cast<GLuint>(batch->MeshType()));
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		glUniform3ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ);
		glUniform1ui(U_MAX_CACHE_SIZE, static_cast<GLuint>(cache.voxelNormPos.Capacity()));
		glUniform1ui(U_IS_MIP, static_cast<GLuint>(isMip[i]));
		glDispatchCompute(voxDimX + 7 / 8, voxDimY + 7 / 8, voxDimZ + 7 / 8);
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		// Voxelization Done!
	}
	//glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
	glEndQuery(GL_TIME_ELAPSED);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	GLuint64 timeElapsed = 0;
	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);
	cache.objectGridInfo.RecieveData(batch->DrawCount());
	cache.voxelCacheUsageSize.RecieveData(1);
	cache.batchVoxCacheCount = 0;
	for(int i = 0; i < batch->DrawCount(); i++)
		cache.batchVoxCacheCount += cache.objectGridInfo.CPUData()[i].voxCount;
	assert(cache.voxelCacheUsageSize.CPUData()[0] == cache.batchVoxCacheCount);

	// Check if we exceeded the max (normally we didnt write bu we incremented counter)
	cache.batchVoxCacheCount = std::min(cache.batchVoxCacheCount, 
										static_cast<uint32_t>(cache.voxelNormPos.Capacity()));
	cache.batchVoxCacheSize = static_cast<double>(cache.batchVoxCacheCount *
												  (sizeof(CVoxelNormPos) +
												  sizeof(VoxelRenderData) +
												  sizeof(CVoxelIds))) /
												  1024.0 /
												  1024.0;

	double time = timeElapsed / 1000000.0;
	GI_LOG("------------------------------------");
	GI_LOG("Voxelization Complete");
	GI_LOG("Cascade Parameters: ");
	GI_LOG("\tObject Span Multiplier: %d", minSpanMultiplier);
	GI_LOG("\tGrid Span %f", gridSpan);
	GI_LOG("Scene Voxelization Time: %f ms", time);
	GI_LOG("Total Vox : %d", cache.batchVoxCacheCount);
	GI_LOG("Total Vox Memory: %f MB", cache.batchVoxCacheSize);
	GI_LOG("------------------------------------");

	glDeleteQueries(1, &queryID);
	return time;
	return 0.0;
}

void ThesisSolution::LinkCacheWithVoxScene(GICudaVoxelScene& scene, 
										   SceneVoxCache& cache,
										   float coverageRatio)
{
	// Send it to CUDA
	Array32<MeshBatchI*> batches = currentScene->getBatches();
	assert(batches.length == cache.cache.size());
	for(unsigned int i = 0; i < cache.cache.size(); i++)
	{
		scene.LinkOGL(batches.arr[i]->getDrawBuffer().getAABBBuffer().getGLBuffer(),
					  batches.arr[i]->getDrawBuffer().getModelTransformBuffer().getGLBuffer(),
					  batches.arr[i]->getDrawBuffer().getModelTransformIndexBuffer().getGLBuffer(),
					  cache.cache[i].objectGridInfo.getGLBuffer(),
					  cache.cache[i].voxelNormPos.getGLBuffer(),
					  cache.cache[i].voxelIds.getGLBuffer(),
					  cache.cache[i].voxelRenderData.getGLBuffer(),
					  static_cast<uint32_t>(batches.arr[i]->DrawCount()),
					  cache.cache[i].batchVoxCacheCount);
	}
	// Allocate at least all of the scene voxel
	scene.AllocateWRTLinkedData(coverageRatio);
}

void ThesisSolution::LevelIncrement()
{
	svoRenderLevel++;
	svoRenderLevel = std::min(svoRenderLevel, voxelOctree.SVOConsts().totalDepth);
	//GI_LOG("Level %d", svoRenderLevel);
}

void ThesisSolution::LevelDecrement()
{
	svoRenderLevel--;
	svoRenderLevel = std::max(svoRenderLevel, voxelOctree.SVOConsts().denseDepth);
	//GI_LOG("Level %d", svoRenderLevel);
}

void ThesisSolution::TraceTypeInc()
{
	traceType++;
	//GI_LOG("Trace Type %d", traceType % 3);
}

void ThesisSolution::TraceTypeDec()
{
	traceType--;
	//GI_LOG("Trace Type %d", traceType % 3);
}

void ThesisSolution::DebugRenderVoxelCache(const Camera& camera, 
										   SceneVoxCache& cache)
{
	//DEBUG VOXEL RENDER
	// Frame Viewport
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(camera.width),
			   static_cast<GLsizei>(camera.height));

	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	glDisable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(true);
	glColorMask(true, true, true, true);

	glClear(GL_COLOR_BUFFER_BIT |
			GL_DEPTH_BUFFER_BIT);

	// Debug Voxelize Scene
	Shader::Unbind(ShaderType::GEOMETRY);
	vertexDebugVoxel.Bind();
	glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(traceType % 2));
	fragmentDebugVoxel.Bind();

	cameraTransform.Bind();
	cameraTransform.Update(camera.generateTransform());

	Array32<MeshBatchI*> batches = currentScene->getBatches();
	for(unsigned int i = 0; i < cache.cache.size(); i++)
	{
		cache.cache[i].objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
		
		DrawBuffer& dBuffer = batches.arr[i]->getDrawBuffer();
		dBuffer.getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
		dBuffer.getModelTransformBuffer().BindAsShaderStorageBuffer(LU_MTRANSFORM);
		dBuffer.getModelTransformIndexBuffer().BindAsShaderStorageBuffer(LU_MTRANSFORM_INDEX);

		cache.cache[i].voxelVAO.Bind();
		cache.cache[i].voxelVAO.Draw(cache.cache[i].batchVoxCacheCount, 0);
	}

	
}

void ThesisSolution::DebugRenderVoxelPage(const Camera& camera, 
										  VoxelDebugVAO& pageVoxels,
										  const CVoxelGrid& voxGrid,
										  uint32_t offset,
										  uint32_t voxCount)
{
	//DEBUG VOXEL RENDER
	// Frame Viewport
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(camera.width),
			   static_cast<GLsizei>(camera.height));
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	glDisable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LESS);
	glDepthMask(true);
	glColorMask(true, true, true, true);

	// Debug Voxelize Pages
	// User World Render Vertex Shader
	Shader::Unbind(ShaderType::GEOMETRY);
	vertexDebugWorldVoxel.Bind();
	glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(traceType % 2));
	fragmentDebugVoxel.Bind();

	// We need grid info buffer as uniform and frame transform buffer
	cameraTransform.Bind();
	cameraTransform.Update(camera.generateTransform());

	VoxelGridInfoGL voxelGridGL = 
	{
		{voxGrid.position.x, voxGrid.position.y, voxGrid.position.z, voxGrid.span},
		{voxGrid.dimension.x, voxGrid.dimension.y, voxGrid.dimension.z, voxGrid.depth},
	};
	gridInfoBuffer.CPUData()[0] = voxelGridGL;
	gridInfoBuffer.SendData();
	gridInfoBuffer.BindAsUniformBuffer(U_VOXEL_GRID_INFO);

	pageVoxels.Bind();
	pageVoxels.Draw(voxCount, offset);
}

double ThesisSolution::DebugRenderSVO(const Camera& camera)
{
	GLuint colorTex = dRenderer.GetGBuffer().getColorGL();

	// Update FrameTransform Matrices 
	// And its inverse realted buffer
	dRenderer.RefreshInvFTransform(camera);
	dRenderer.GetFTransform().Update(camera.generateTransform());
	
	//DEBUG
	SVOTraceType traceTypeEnum = static_cast<SVOTraceType>(traceType % 3);

	// Raytrace voxel scene
	double time;
	time = voxelOctree.DebugTraceSVO(colorTex,
									 dRenderer.GetInvFTransfrom(),
									 dRenderer.GetFTransform(),
									 {DeferredRenderer::gBuffWidth,
									  DeferredRenderer::gBuffHeight},
									  svoRenderLevel,
									  traceTypeEnum);

	// Tell deferred renderer to post process color buffer;
	dRenderer.ShowColorGBuffer(camera);
	return time;
}

void ThesisSolution::Frame(const Camera& mainRenderCamera)
{
	// Zero out debug transfer time since it may not be used
	debugVoxTransferTime = 0;

	// VoxelSceneUpdate
	double ioTimeSegment, transformTimeSegment;
	ioTime = 0;
	transformTime = 0;
	svoTime = 0;

	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		voxelScenes[i].MapGLPointers();

		// Cascade #1 Update
		voxelScenes[i].VoxelUpdate(ioTimeSegment,
								   transformTimeSegment,
								   mainRenderCamera.pos,
								   static_cast<float>(0x1 << (3 - i)));
		ioTime += ioTimeSegment;
		transformTime += transformTimeSegment;
	}
	ioTime += ioTimeSegment;
	transformTime += transformTimeSegment;
	
	// Octree Update
	svoTime = voxelOctree.UpdateSVO();

	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		voxelScenes[i].UnmapGLPointers();
	}

	// Voxel Count in Pages
	for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
	{
		voxelCaches[i].voxOctreeCount = voxelScenes[i].VoxelCountInPage();
		voxelCaches[i].voxOctreeSize = static_cast<double>(voxelCaches[i].voxOctreeCount * sizeof(uint32_t) * 4) / 1024 / 1024;
	}

	// Here check TW Bar if user wants to render voxels
	switch(renderScheme)
	{
		case GI_DEFERRED:
		{
			dRenderer.Render(*currentScene, mainRenderCamera);
			break;
		}
		case GI_LIGHT_INTENSITY:
		{
			glClearColor(1.0f, 1.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			break;
		}		
		case GI_SVO_LEVELS:
		{
			// Start Render
			glClearColor(1.0f, 1.0f, 0.0f, 0.0f);
			debugVoxTransferTime = DebugRenderSVO(mainRenderCamera);
			break;
		}
		case GI_VOXEL_PAGE:
		{
			unsigned int totalVoxCount = 0;
			for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
				totalVoxCount += voxelCaches[i].voxOctreeCount;

			voxelNormPosBuffer.Resize(totalVoxCount);
			voxelColorBuffer.Resize(totalVoxCount);

			// Cuda Register	
			CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vaoNormPosResource, 
												    voxelNormPosBuffer.getGLBuffer(), 
													cudaGraphicsMapFlagsWriteDiscard));
			CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vaoRenderResource, 
													voxelColorBuffer.getGLBuffer(), 
													cudaGraphicsMapFlagsWriteDiscard));

			CVoxelNormPos* dVoxNormPos = nullptr;
			uchar4* dVoxColor = nullptr;
			size_t bufferSize;
			
			CUDA_CHECK(cudaGraphicsMapResources(1, &vaoNormPosResource));
			CUDA_CHECK(cudaGraphicsMapResources(1, &vaoRenderResource));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dVoxNormPos), 
															&bufferSize,
															vaoNormPosResource));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dVoxColor),
															&bufferSize,
															vaoRenderResource));

			std::vector<uint32_t> offsets;
			std::vector<uint32_t> counts;

			// Create VAO after resize since buffer id can change
			VoxelDebugVAO vao(voxelNormPosBuffer, voxelColorBuffer);

			// Start Render
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT |
					GL_DEPTH_BUFFER_BIT);
			
			
			uint32_t voxelCount = 0, voxelOffset = 0;
			std::vector<CVoxelGrid> voxGrids(GI_CASCADE_COUNT);
			
			debugVoxTransferTime = 0;
			for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
			{
				offsets.push_back(voxelOffset);
				debugVoxTransferTime += voxelScenes[i].VoxDataToGL(dVoxNormPos + voxelOffset,
																   dVoxColor + voxelOffset,
																   voxGrids[i],
																   voxelCount,
																   voxelCaches[i].voxOctreeCount);
				voxelOffset += voxelCount;
				counts.push_back(voxelCount);
			}
			// All written unmap
			CUDA_CHECK(cudaGraphicsUnmapResources(1, &vaoNormPosResource));
			CUDA_CHECK(cudaGraphicsUnmapResources(1, &vaoRenderResource));
			CUDA_CHECK(cudaGraphicsUnregisterResource(vaoNormPosResource));
			CUDA_CHECK(cudaGraphicsUnregisterResource(vaoRenderResource));

			// Render
			for(unsigned int i = 0; i < GI_CASCADE_COUNT; i++)
			{
				DebugRenderVoxelPage(mainRenderCamera, 
									 vao, 
									 voxGrids[i], offsets[i], counts[i]);
			}
			break;
		}
		case GI_VOXEL_CACHE2048:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, voxelCaches[0]);
			break;
		}
		case GI_VOXEL_CACHE1024:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, voxelCaches[1]);
			break;
		}
		case GI_VOXEL_CACHE512:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, voxelCaches[2]);
			break;
		}
	}
}

void ThesisSolution::SetFPS(double fpsMS)
{
	frameTime = fpsMS;
}

void ThesisSolution::LevelIncrement(void* solPtr)
{
	static_cast<ThesisSolution*>(solPtr)->LevelIncrement();
}
void ThesisSolution::LevelDecrement(void* solPtr)
{
	static_cast<ThesisSolution*>(solPtr)->LevelDecrement();
}

void ThesisSolution::TraceIncrement(void* solutionPtr)
{
	static_cast<ThesisSolution*>(solutionPtr)->TraceTypeInc();
}

void ThesisSolution::TraceDecrement(void* solutionPtr)
{
	static_cast<ThesisSolution*>(solutionPtr)->TraceTypeDec();
}