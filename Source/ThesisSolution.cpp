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
const size_t ThesisSolution::MaxVoxelCacheSize2048 = static_cast<size_t>(1024 * 1024 * 1.5f);
const size_t ThesisSolution::MaxVoxelCacheSize1024 = static_cast<size_t>(1024 * 1024 * 2.0f);
const size_t ThesisSolution::MaxVoxelCacheSize512 = static_cast<size_t>(1024 * 1024 * 1.5f);

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
	, cache2048(InitialObjectGridSize, MaxVoxelCacheSize2048)
	, cache1024(InitialObjectGridSize, MaxVoxelCacheSize1024)
	, cache512(InitialObjectGridSize, MaxVoxelCacheSize512)
	, bar(nullptr)
	, voxelScene2048(intialCamPos, CascadeSpan, CascadeDim)
	, voxelScene1024(intialCamPos, CascadeSpan * 2, CascadeDim)
	, voxelScene512(intialCamPos, CascadeSpan * 4, CascadeDim)
	, renderScheme(GI_VOXEL_PAGE)
	//, renderScheme(GI_DEFERRED)
	, gridInfoBuffer(1)
	, voxelNormPosBuffer(512)
	, voxelColorBuffer(512)
	, voxelOctree()
{
	renderType = TwDefineEnum("RenderType", renderSchemeVals, GI_END);
	gridInfoBuffer.AddData({});
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
	voxelScene1024.Reset();
	voxelScene2048.Reset();
	voxelScene512.Reset();

	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	currentScene = &s;

	// Voxelization
	// and Voxel Cache Creation
	double voxelTotaltime = 0.0;
	voxelTotaltime += Voxelize(cache2048, CascadeSpan, 1, true);
	voxelTotaltime += Voxelize(cache1024, CascadeSpan * 2, 2, false);
	voxelTotaltime += Voxelize(cache512, CascadeSpan * 4, 4, false);
	GI_LOG("Scene voxelization completed. Elapsed time %f ms", voxelTotaltime);

	// Voxel Page System Linking
	LinkCacheWithVoxScene(voxelScene2048, cache2048, 1.0f);
	LinkCacheWithVoxScene(voxelScene1024, cache1024, 1.0f);
	LinkCacheWithVoxScene(voxelScene512, cache512, 1.0f);

	// Allocators Link
	GICudaAllocator* allocators[] = 
	{
		voxelScene512.Allocator(),
		voxelScene1024.Allocator(),
		voxelScene2048.Allocator(),
	};
	voxelOctree.LinkAllocators(allocators, 3, 
							   currentScene->SVOTotalSize(),
							   currentScene->SVOLevelSizes());
	svoRenderLevel = voxelOctree.SVOConsts().totalDepth;

	// Memory Usage Total
	GI_LOG("Voxel Sytem #1 Total Memory Usage %f MB", 
		   static_cast<double>(voxelScene2048.AllocatorMemoryUsage()) / 1024.0 / 1024.0);
	GI_LOG("Voxel Sytem #2 Total Memory Usage %f MB",
		   static_cast<double>(voxelScene1024.AllocatorMemoryUsage()) / 1024.0 / 1024.0);
	GI_LOG("Voxel Sytem #3 Total Memory Usage %f MB",
		   static_cast<double>(voxelScene512.AllocatorMemoryUsage()) / 1024.0 / 1024.0);
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
	TwAddVarRO(bar, "voxCache512", TW_TYPE_UINT32, &cache2048.voxInfo.sceneVoxCacheCount,
			   " label='Cascade#1 Count' group='Voxel Cache' help='Cache voxel count.' ");
	TwAddVarRO(bar, "voxCacheSize512", TW_TYPE_DOUBLE, &cache2048.voxInfo.sceneVoxCacheSize,
			   " label='Cascade#1 Size(MB)' group='Voxel Cache' precision=2 help='Cache voxel total size in megabytes.' ");
	TwAddVarRO(bar, "voxCache256", TW_TYPE_UINT32, &cache1024.voxInfo.sceneVoxCacheCount,
			   " label='Cascade#2 Count' group='Voxel Cache' help='Cache voxel count.' ");
	TwAddVarRO(bar, "voxCacheSize256", TW_TYPE_DOUBLE, &cache1024.voxInfo.sceneVoxCacheSize,
			   " label='Cascade#2 Size(MB)' group='Voxel Cache' precision=2 help='Cache voxel total size in megabytes.' ");
	TwAddVarRO(bar, "voxCache128", TW_TYPE_UINT32, &cache512.voxInfo.sceneVoxCacheCount,
			   " label='Cascade#3 Count' group='Voxel Cache' help='Cache voxel count.' ");
	TwAddVarRO(bar, "voxCacheSize128", TW_TYPE_DOUBLE, &cache512.voxInfo.sceneVoxCacheSize,
			   " label='Cascade#3 Size(MB)' group='Voxel Cache' precision=2 help='Cache voxel total size in megabytes.' ");
	TwAddSeparator(bar, NULL, NULL);
	TwAddVarRO(bar, "voxUsed512", TW_TYPE_UINT32, &cache2048.voxInfo.sceneVoxOctreeCount,
			   " label='Cascade#1 Count' group='Voxel Octree' help='Voxel count in octree.' ");
	TwAddVarRO(bar, "voxUsedSize512", TW_TYPE_DOUBLE, &cache2048.voxInfo.sceneVoxOctreeSize,
			   " label='Cascade#1 Size(MB)' group='Voxel Octree' precision=2 help='Octree Voxel total size in megabytes.' ");
	TwAddVarRO(bar, "voxUsed256", TW_TYPE_UINT32, &cache1024.voxInfo.sceneVoxOctreeCount,
			   " label='Cascade#2 Count' group='Voxel Octree' help='Voxel count in octree.' ");
	TwAddVarRO(bar, "voxUsedSize256", TW_TYPE_DOUBLE, &cache1024.voxInfo.sceneVoxOctreeSize,
			   " label='Cascade#2 Size(MB)' group='Voxel Octree' precision=2 help='Octree Voxel total size in megabytes.' ");
	TwAddVarRO(bar, "voxUsed128", TW_TYPE_UINT32, &cache512.voxInfo.sceneVoxOctreeCount,
			   " label='Cascade#3 Count' group='Voxel Octree' help='Voxel count in octree.' ");
	TwAddVarRO(bar, "voxUsedSize128", TW_TYPE_DOUBLE, &cache512.voxInfo.sceneVoxOctreeSize,
			   " label='Cascade#3 Size(MB)' group='Voxel Octree' precision=2 help='Octree Voxel total size in megabytes.' ");
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
								float gridSpan, unsigned int minSpanMultiplier,
								bool isInnerCascade)
{
	cache.objectGridInfo.Resize(currentScene->DrawCount());

	// Timing Voxelization Process
	GLuint queryID;
	glGenQueries(1, &queryID);
	glBeginQuery(GL_TIME_ELAPSED, queryID);

	//
	glClear(GL_COLOR_BUFFER_BIT);
	DrawBuffer& dBuffer = currentScene->getDrawBuffer();
	VoxelRenderTexture voxelRenderTexture;

	// Determine Voxel Sizes
	computeDetermineVoxSpan.Bind();
	cache.objectGridInfo.Resize(currentScene->DrawCount());
	currentScene->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
	cache.objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	glUniform1ui(U_TOTAL_OBJ_COUNT, static_cast<GLuint>(currentScene->DrawCount()));
	glUniform1f(U_MIN_SPAN, currentScene->MinSpan());
	glUniform1ui(U_MAX_GRID_DIM, VOXEL_GRID_SIZE);

	size_t blockCount = (currentScene->DrawCount() / 128);
	size_t factor = ((currentScene->DrawCount() % 128) == 0) ? 0 : 1;
	blockCount += factor;
	glDispatchCompute(static_cast<GLuint>(blockCount), 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	cache.objectGridInfo.RecieveData(currentScene->DrawCount());

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
	std::vector<GLuint> isMip(currentScene->DrawCount(), 0);
	for(unsigned int i = 0; i < isMip.size(); i++)
	{
		if(cache.objectGridInfo.CPUData()[i].span < currentScene->MinSpan() * minSpanMultiplier)
		{
			isMip[i] = (isInnerCascade) ? 0 : 1;
			cache.objectGridInfo.CPUData()[i].span = currentScene->MinSpan() * minSpanMultiplier;
		}
			
	}
	cache.objectGridInfo.SendData();

	// For Each Object
	voxelRenderTexture.Clear();
	for(unsigned int i = 0; i < currentScene->DrawCount(); i++)
	{
		// Skip objects that cant fit
		if(cache.objectGridInfo.CPUData()[i].span != currentScene->MinSpan() * minSpanMultiplier)
			continue;

		//
		//unsigned int isMip = 

		// First Call Voxelize over 3D Texture
		currentScene->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
		voxelRenderTexture.BindAsImage(I_VOX_WRITE, GL_WRITE_ONLY);
		vertexVoxelizeObject.Bind();
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		geomVoxelizeObject.Bind();
		fragmentVoxelizeObject.Bind();
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		currentScene->getGPUBuffer().Bind();

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
		const AABBData& objAABB = currentScene->getDrawBuffer().getAABBBuffer().CPUData()[i];
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
		glUniform1ui(U_OBJ_TYPE, static_cast<GLuint>(VoxelObjectType::STATIC));
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
	cache.objectGridInfo.RecieveData(currentScene->DrawCount());
	cache.voxelCacheUsageSize.RecieveData(1);
	cache.voxInfo.sceneVoxCacheCount = 0;
	for(int i = 0; i < currentScene->DrawCount(); i++)
		cache.voxInfo.sceneVoxCacheCount += cache.objectGridInfo.CPUData()[i].voxCount;
	assert(cache.voxelCacheUsageSize.CPUData()[0] == cache.voxInfo.sceneVoxCacheCount);
	cache.voxInfo.sceneVoxCacheSize = static_cast<double>(cache.voxInfo.sceneVoxCacheCount * 
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
	GI_LOG("Total Vox : %d", cache.voxInfo.sceneVoxCacheCount);
	GI_LOG("Total Vox Memory: %f MB", cache.voxInfo.sceneVoxCacheSize);
	GI_LOG("------------------------------------");

	glDeleteQueries(1, &queryID);
	return time;
}

void ThesisSolution::LinkCacheWithVoxScene(GICudaVoxelScene& scene, 
										   VoxelObjectCache& cache,
										   float coverageRatio)
{
	// Send it to CUDA
	scene.LinkOGL(currentScene->getDrawBuffer().getAABBBuffer().getGLBuffer(),
						  currentScene->getDrawBuffer().getModelTransformBuffer().getGLBuffer(),
						  cache.objectGridInfo.getGLBuffer(),
						  cache.voxelNormPos.getGLBuffer(),
						  cache.voxelIds.getGLBuffer(),
						  cache.voxelRenderData.getGLBuffer(),
						  static_cast<uint32_t>(currentScene->DrawCount()),
						  cache.voxInfo.sceneVoxCacheCount);
	// Allocate at least all of the scene voxel
	scene.AllocateWRTLinkedData(coverageRatio);
}

void ThesisSolution::LevelIncrement()
{
	svoRenderLevel++;
	svoRenderLevel = std::min(svoRenderLevel, voxelOctree.SVOConsts().totalDepth);
}

void ThesisSolution::LevelDecrement()
{
	svoRenderLevel--;
	svoRenderLevel = std::max(svoRenderLevel, voxelOctree.SVOConsts().denseDepth);
}


void ThesisSolution::DebugRenderVoxelCache(const Camera& camera, VoxelObjectCache& cache)
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
	fragmentDebugVoxel.Bind();

	cameraTransform.Bind();
	cameraTransform.Update(camera.generateTransform());

	cache.objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	currentScene->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);

	// Bind Model Transform
	DrawBuffer& dBuffer = currentScene->getDrawBuffer();
	dBuffer.getModelTransformBuffer().BindAsShaderStorageBuffer(LU_MTRANSFORM);

	cache.voxelVAO.Bind();
	cache.voxelVAO.Draw(cache.voxInfo.sceneVoxCacheCount, 0);
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
	
	// Raytrace voxel scene
	double time;
	time = voxelOctree.DebugTraceSVO(colorTex,
									 dRenderer.GetInvFTransfrom(),
									 dRenderer.GetFTransform(),
									 {DeferredRenderer::gBuffWidth,
									  DeferredRenderer::gBuffHeight},
									  svoRenderLevel);

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

	voxelScene2048.MapGLPointers();
	voxelScene1024.MapGLPointers();
	voxelScene512.MapGLPointers();

	// Cascade #1 Update
	voxelScene2048.VoxelUpdate(ioTimeSegment,
							  transformTimeSegment,
							  mainRenderCamera.pos,
							  4.0f);
	ioTime += ioTimeSegment;
	transformTime += transformTimeSegment;

	// Cascade #2 Update
	voxelScene1024.VoxelUpdate(ioTimeSegment,
							  transformTimeSegment,
							  mainRenderCamera.pos,
							  2.0f);
	ioTime += ioTimeSegment;
	transformTime += transformTimeSegment;

	// Cascade #3 Update
	voxelScene512.VoxelUpdate(ioTimeSegment,
							  transformTimeSegment,
							  mainRenderCamera.pos,
							  1.0f);
	ioTime += ioTimeSegment;
	transformTime += transformTimeSegment;
	
	// Octree Update
	svoTime = voxelOctree.UpdateSVO();

	voxelScene2048.UnmapGLPointers();
	voxelScene1024.UnmapGLPointers();
	voxelScene512.UnmapGLPointers();

	// Voxel Count in Pages
	cache2048.voxInfo.sceneVoxOctreeCount = voxelScene2048.VoxelCountInPage();
	cache2048.voxInfo.sceneVoxOctreeSize = static_cast<double>(cache2048.voxInfo.sceneVoxOctreeCount * sizeof(uint32_t) * 4) / 1024 / 1024;

	cache1024.voxInfo.sceneVoxOctreeCount = voxelScene1024.VoxelCountInPage();
	cache1024.voxInfo.sceneVoxOctreeSize = static_cast<double>(cache1024.voxInfo.sceneVoxOctreeCount * sizeof(uint32_t) * 4) / 1024 / 1024;

	cache512.voxInfo.sceneVoxOctreeCount = voxelScene512.VoxelCountInPage();
	cache512.voxInfo.sceneVoxOctreeSize = static_cast<double>(cache512.voxInfo.sceneVoxOctreeCount * sizeof(uint32_t) * 4) / 1024 / 1024;
	
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
			unsigned int totalVoxCount = cache2048.voxInfo.sceneVoxOctreeCount +
										 cache1024.voxInfo.sceneVoxOctreeCount +
										 cache512.voxInfo.sceneVoxOctreeCount;
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
			CVoxelGrid voxGrid512;
			CVoxelGrid voxGrid1024;
			CVoxelGrid voxGrid2048;

			debugVoxTransferTime = 0;
			// 512
			offsets.push_back(voxelOffset);
			debugVoxTransferTime += voxelScene512.VoxDataToGL(dVoxNormPos + voxelOffset,
															  dVoxColor + voxelOffset,
															  voxGrid512,
															  voxelCount,
															  cache512.voxInfo.sceneVoxOctreeCount);
			voxelOffset += voxelCount;
			counts.push_back(voxelCount);

			// 1024
			offsets.push_back(voxelOffset);
			debugVoxTransferTime += voxelScene1024.VoxDataToGL(dVoxNormPos + voxelOffset,
															   dVoxColor + voxelOffset,
															   voxGrid1024,
															   voxelCount,
															   cache1024.voxInfo.sceneVoxOctreeCount);
			voxelOffset += voxelCount;
			counts.push_back(voxelCount);

			// 2048
			offsets.push_back(voxelOffset);
			debugVoxTransferTime += voxelScene2048.VoxDataToGL(dVoxNormPos + voxelOffset,
															   dVoxColor + voxelOffset,
															   voxGrid2048,
															   voxelCount,
															   cache2048.voxInfo.sceneVoxOctreeCount);
			voxelOffset += voxelCount;
			counts.push_back(voxelCount);

			// All written unmap
			CUDA_CHECK(cudaGraphicsUnmapResources(1, &vaoNormPosResource));
			CUDA_CHECK(cudaGraphicsUnmapResources(1, &vaoRenderResource));
			CUDA_CHECK(cudaGraphicsUnregisterResource(vaoNormPosResource));
			CUDA_CHECK(cudaGraphicsUnregisterResource(vaoRenderResource));

			// Render
			DebugRenderVoxelPage(mainRenderCamera, vao, voxGrid512, offsets[0], counts[0]);
			DebugRenderVoxelPage(mainRenderCamera, vao, voxGrid1024, offsets[1], counts[1]);
			DebugRenderVoxelPage(mainRenderCamera, vao, voxGrid2048, offsets[2], counts[2]);	
			break;
		}
		case GI_VOXEL_CACHE2048:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, cache2048);
			break;
		}
		case GI_VOXEL_CACHE1024:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, cache1024);
			break;
		}
		case GI_VOXEL_CACHE512:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, cache512);
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