#include "ThesisSolution.h"
#include "Globals.h"
#include "SceneI.h"
#include "DrawBuffer.h"
#include "Macros.h"
#include "Camera.h"
#include "DeferredRenderer.h"
#include "SceneLights.h"

size_t ThesisSolution::InitialObjectGridSize = 256;
size_t ThesisSolution::MaxVoxelCacheSize512 = static_cast<size_t>(1024 * 1024 * 3.5f);
size_t ThesisSolution::MaxVoxelCacheSize256 = static_cast<size_t>(1024 * 1024 * 2.5f);
size_t ThesisSolution::MaxVoxelCacheSize128 = static_cast<size_t>(1024 * 1024 * 1.5f);

const TwEnumVal ThesisSolution::renderSchemeVals[] = 
{ 
	{ GI_DEFERRED, "Deferred" }, 
	{ GI_LIGHT_INTENSITY, "LI Buffer Only"}, 
	{ GI_VOXEL_PAGE, "Render Voxel Page" },
	{ GI_VOXEL_CACHE512, "Render Voxel Cache 512" },
	{ GI_VOXEL_CACHE256, "Render Voxel Cache 256" },
	{ GI_VOXEL_CACHE128, "Render Voxel Cache 128" }
};

ThesisSolution::ThesisSolution(DeferredRenderer& dRenderer, const IEVector3& intialCamPos)
	: currentScene(nullptr)
	, dRenderer(dRenderer)
	, vertexDebugVoxel(ShaderType::VERTEX, "Shaders/VoxRender.vert")
	, vertexDebugWorldVoxel(ShaderType::VERTEX, "Shaders/VoxRenderWorld.vert")
	, fragmentDebugVoxel(ShaderType::FRAGMENT, "Shaders/VoxRender.frag")
	, vertexDebugWorldVoxelCascade(ShaderType::VERTEX, "Shaders/VoxRenderWorldCascaded.vert")
	, fragmentDebugWorldVoxelCascade(ShaderType::FRAGMENT, "Shaders/VoxRenderWorldCascaded.frag")
	, vertexVoxelizeObject(ShaderType::VERTEX, "Shaders/VoxelizeGeom.vert")
	, geomVoxelizeObject(ShaderType::GEOMETRY, "Shaders/VoxelizeGeom.geom")
	, fragmentVoxelizeObject(ShaderType::FRAGMENT, "Shaders/VoxelizeGeom.frag")
	, computeVoxelizeCount(ShaderType::COMPUTE, "Shaders/VoxelizeGeomCount.glsl")
	, computePackObjectVoxels(ShaderType::COMPUTE, "Shaders/PackObjectVoxels.glsl")
	, computeDetermineVoxSpan(ShaderType::COMPUTE, "Shaders/DetermineVoxSpan.glsl")
	, cache512(InitialObjectGridSize, MaxVoxelCacheSize512)
	, cache256(InitialObjectGridSize, MaxVoxelCacheSize256)
	, cache128(InitialObjectGridSize, MaxVoxelCacheSize128)
	, bar(nullptr)
	, voxelScene512(intialCamPos, 0.513f, 512)
	, voxelScene256(intialCamPos, 0.513f * 2, 512)
	, voxelScene128(intialCamPos, 0.513f * 4, 512)
	, renderScheme(GI_VOXEL_PAGE)
	//, renderScheme(GI_VOXEL_CACHE512)
	, gridInfoBuffer(1)
{

	voxelScene512.LinkDeferredRendererBuffers(dRenderer.GetGBuffer().getDepthGLView(),
											  dRenderer.GetGBuffer().getNormalGL(),
											  dRenderer.GetLightIntensityBufferGL());
	voxelScene256.LinkDeferredRendererBuffers(dRenderer.GetGBuffer().getDepthGLView(),
											  dRenderer.GetGBuffer().getNormalGL(),
											  dRenderer.GetLightIntensityBufferGL());
	voxelScene128.LinkDeferredRendererBuffers(dRenderer.GetGBuffer().getDepthGLView(),
											  dRenderer.GetGBuffer().getNormalGL(),
											  dRenderer.GetLightIntensityBufferGL());
	renderType = TwDefineEnum("RenderType", renderSchemeVals, 6);
	gridInfoBuffer.AddData({});
}

ThesisSolution::~ThesisSolution()
{
	voxelScene512.UnLinkDeferredRendererBuffers();
	voxelScene256.UnLinkDeferredRendererBuffers();
	voxelScene128.UnLinkDeferredRendererBuffers();
}

bool ThesisSolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}

void ThesisSolution::Init(SceneI& s)
{
	// Reset GICudaScene
	voxelScene512.Reset();
	voxelScene256.Reset();
	voxelScene128.Reset();

	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	currentScene = &s;

	// Voxelization
	// and Voxel Cache Creation
	double voxelTotaltime = 0.0;
	voxelTotaltime += Voxelize(cache512, 0.513f, 1);
	voxelTotaltime += Voxelize(cache256, 0.513f * 2, 2);
	voxelTotaltime += Voxelize(cache128, 0.513f * 4, 4);
	GI_LOG("Scene voxelization completed. Elapsed time %f ms", voxelTotaltime);

	// Voxel Page System Linking
	LinkCacheWithVoxScene(voxelScene512, cache512);
	LinkCacheWithVoxScene(voxelScene256, cache256);
	LinkCacheWithVoxScene(voxelScene128, cache128);
	
	// Memory Usage Total
	GI_LOG("Voxel Sytem #1 Total Memory Usage %f MB", 
		   static_cast<double>(voxelScene512.AllocatorMemoryUsage()) / 1024.0 / 1024.0);
	GI_LOG("Voxel Sytem #2 Total Memory Usage %f MB",
		   static_cast<double>(voxelScene256.AllocatorMemoryUsage()) / 1024.0 / 1024.0);
	GI_LOG("Voxel Sytem #3 Total Memory Usage %f MB",
		   static_cast<double>(voxelScene128.AllocatorMemoryUsage()) / 1024.0 / 1024.0);


	// Tw Bar Creation
	bar = TwNewBar("ThesisGI");
	TwDefine(" ThesisGI refresh=0.01 ");

	// FPS Show
	TwAddVarRO(bar, "fTime", TW_TYPE_DOUBLE, &frameTime,
			   " label='Frame(ms)' precision=2 help='Frame Time in milliseconds.' ");
	TwAddVarRW(bar, "rendering", renderType,
			   &renderScheme,
			   " label='Render' help='Change What to show on screen' ");
	TwAddVarRW(bar, "giOn", TW_TYPE_BOOLCPP,
			   &giOn,
			   " label='GI On' help='Cone Tracing GI On off' ");
	TwAddSeparator(bar, NULL, NULL);
	TwAddVarRO(bar, "voxCache512", TW_TYPE_UINT32, &cache512.voxInfo.sceneVoxCacheCount,
			   " label='Cascade#1 Count' group='Voxel Cache' help='Cache voxel count.' ");
	TwAddVarRO(bar, "voxCacheSize512", TW_TYPE_DOUBLE, &cache512.voxInfo.sceneVoxCacheSize,
			   " label='Cascade#1 Size(MB)' group='Voxel Cache' precision=2 help='Cache voxel total size in megabytes.' ");
	TwAddVarRO(bar, "voxCache256", TW_TYPE_UINT32, &cache256.voxInfo.sceneVoxCacheCount,
			   " label='Cascade#2 Count' group='Voxel Cache' help='Cache voxel count.' ");
	TwAddVarRO(bar, "voxCacheSize256", TW_TYPE_DOUBLE, &cache256.voxInfo.sceneVoxCacheSize,
			   " label='Cascade#2 Size(MB)' group='Voxel Cache' precision=2 help='Cache voxel total size in megabytes.' ");
	TwAddVarRO(bar, "voxCache128", TW_TYPE_UINT32, &cache128.voxInfo.sceneVoxCacheCount,
			   " label='Cascade#3 Count' group='Voxel Cache' help='Cache voxel count.' ");
	TwAddVarRO(bar, "voxCacheSize128", TW_TYPE_DOUBLE, &cache128.voxInfo.sceneVoxCacheSize,
			   " label='Cascade#3 Size(MB)' group='Voxel Cache' precision=2 help='Cache voxel total size in megabytes.' ");
	TwAddSeparator(bar, NULL, NULL);
	TwAddVarRO(bar, "voxUsed512", TW_TYPE_UINT32, &cache512.voxInfo.sceneVoxOctreeCount,
			   " label='Cascade#1 Count' group='Voxel Octree' help='Voxel count in octree.' ");
	TwAddVarRO(bar, "voxUsedSize512", TW_TYPE_DOUBLE, &cache512.voxInfo.sceneVoxOctreeSize,
			   " label='Cascade#1 Size(MB)' group='Voxel Octree' precision=2 help='Octree Voxel total size in megabytes.' ");
	TwAddVarRO(bar, "voxUsed256", TW_TYPE_UINT32, &cache256.voxInfo.sceneVoxOctreeCount,
			   " label='Cascade#2 Count' group='Voxel Octree' help='Voxel count in octree.' ");
	TwAddVarRO(bar, "voxUsedSize256", TW_TYPE_DOUBLE, &cache256.voxInfo.sceneVoxOctreeSize,
			   " label='Cascade#2 Size(MB)' group='Voxel Octree' precision=2 help='Octree Voxel total size in megabytes.' ");
	TwAddVarRO(bar, "voxUsed128", TW_TYPE_UINT32, &cache128.voxInfo.sceneVoxOctreeCount,
			   " label='Cascade#3 Count' group='Voxel Octree' help='Voxel count in octree.' ");
	TwAddVarRO(bar, "voxUsedSize128", TW_TYPE_DOUBLE, &cache128.voxInfo.sceneVoxOctreeSize,
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
	TwDefine(" ThesisGI size='325 400' ");
	TwDefine(" ThesisGI valueswidth=fit ");
}

void ThesisSolution::Release()
{
	if(bar) TwDeleteBar(bar);
}

double ThesisSolution::Voxelize(VoxelObjectCache& cache,
								float gridSpan, unsigned int minSpanMultiplier)
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
	glUniform1f(U_MIN_SPAN, currentScene->MinSpan() * minSpanMultiplier);
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
	// (teting conservative rendering on maxwell)
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

	// For Each Object
	voxelRenderTexture.Clear();
	for(unsigned int i = 0; i < currentScene->DrawCount(); i++)
	{
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

		// Before Last Call calcuate span ratio for this object
		auto& transformBuffer = currentScene->getDrawBuffer().getModelTransformBuffer().CPUData();
		auto& objGridInfoBuffer = cache.objectGridInfo.CPUData();
		// Extract scale 
		IEVector3 scale = IEMatrix4x4::ExtractScaleInfo(transformBuffer[i].model);
		assert(fabs(scale.getX() - scale.getY()) < 0.0001f);
		assert(fabs(scale.getY() - scale.getZ()) < 0.0001f);
		GLuint spanRatio = static_cast<unsigned int>((objGridInfoBuffer[i].span * scale.getX() + 0.1f) / gridSpan);
		// Fast nearest pow of two
		spanRatio--;
		spanRatio |= spanRatio >> 1;
		spanRatio |= spanRatio >> 2;
		spanRatio |= spanRatio >> 4;
		spanRatio |= spanRatio >> 8;
		spanRatio |= spanRatio >> 16;
		spanRatio++;

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
		glUniform1ui(U_SPAN_RATIO, static_cast<GLuint>(spanRatio));
		glDispatchCompute(voxDimX + 7 / 8, voxDimY + 7 / 8, voxDimZ + 7 / 8);
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		// Voxelization Done!
	}
	//glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
	glEndQuery(GL_TIME_ELAPSED);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glFlush();

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

void ThesisSolution::LinkCacheWithVoxScene(GICudaVoxelScene& scene, VoxelObjectCache& cache)
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
	// Link ShadowMaps and GBuffer textures to cuda
	scene.LinkSceneTextures(currentScene->getSceneLights().GetShadowMapArrayR32F());
	// Allocate at least all of the scene voxel
	scene.AllocateInitialPages(static_cast<uint32_t>(cache.voxInfo.sceneVoxCacheCount));
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
										  bool isOuterCascade,
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

	if(isOuterCascade)
	{
		vertexDebugWorldVoxelCascade.Bind();
		fragmentDebugWorldVoxelCascade.Bind();
	}
	else
	{
		vertexDebugWorldVoxel.Bind();
		fragmentDebugVoxel.Bind();
	}

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
	pageVoxels.Draw(voxCount, 0);
}

void ThesisSolution::Frame(const Camera& mainRenderCamera)
{
	// Zero out debug transfer time since it may not be used
	debugVoxTransferTime = 0;

	// VoxelSceneUpdate
	double ioTimeSegment, transformTimeSegment, svoTimeSegment;
	ioTime = 0;
	transformTime = 0;
	svoTime = 0;

	// Cascade #1 Update
	voxelScene512.VoxelUpdate(ioTimeSegment,
							  transformTimeSegment,
							  svoTimeSegment,
							  mainRenderCamera.pos);
	ioTime += ioTimeSegment;
	transformTime += transformTimeSegment;
	svoTime += svoTimeSegment;

	// Cascade #2 Update
	voxelScene256.VoxelUpdate(ioTimeSegment,
							  transformTimeSegment,
							  svoTimeSegment,
							  mainRenderCamera.pos);
	ioTime += ioTimeSegment;
	transformTime += transformTimeSegment;
	svoTime += svoTimeSegment;

	// Cascade #3 Update
	voxelScene128.VoxelUpdate(ioTimeSegment,
							  transformTimeSegment,
							  svoTimeSegment,
							  mainRenderCamera.pos);
	ioTime += ioTimeSegment;
	transformTime += transformTimeSegment;
	svoTime += svoTimeSegment;

	// Voxel Count in Pages
	cache512.voxInfo.sceneVoxOctreeCount = voxelScene512.VoxelCountInPage();
	cache512.voxInfo.sceneVoxOctreeSize = static_cast<double>(cache512.voxInfo.sceneVoxOctreeCount * sizeof(uint32_t) * 4) / 1024 / 1024;

	cache256.voxInfo.sceneVoxOctreeCount = voxelScene256.VoxelCountInPage();
	cache256.voxInfo.sceneVoxOctreeSize = static_cast<double>(cache256.voxInfo.sceneVoxOctreeCount * sizeof(uint32_t) * 4) / 1024 / 1024;

	cache128.voxInfo.sceneVoxOctreeCount = voxelScene128.VoxelCountInPage();
	cache128.voxInfo.sceneVoxOctreeSize = static_cast<double>(cache128.voxInfo.sceneVoxOctreeCount * sizeof(uint32_t) * 4) / 1024 / 1024;
	
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
			break;
		}		
		case GI_VOXEL_PAGE:
		{
			// Clear the frame since each function will overwrite eachother
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			glClear(GL_COLOR_BUFFER_BIT |
					GL_DEPTH_BUFFER_BIT);

			CVoxelGrid voxGrid128;
			VoxelDebugVAO vao128 = voxelScene128.VoxelDataForRendering(voxGrid128, debugVoxTransferTime, cache128.voxInfo.sceneVoxOctreeCount);
			DebugRenderVoxelPage(mainRenderCamera, vao128, voxGrid128, true,
								 cache128.voxInfo.sceneVoxOctreeCount);

			glClear(GL_DEPTH_BUFFER_BIT);

			CVoxelGrid voxGrid256;
			VoxelDebugVAO vao256 = voxelScene256.VoxelDataForRendering(voxGrid256, debugVoxTransferTime, cache256.voxInfo.sceneVoxOctreeCount);
			DebugRenderVoxelPage(mainRenderCamera, vao256, voxGrid256, true,
								 cache256.voxInfo.sceneVoxOctreeCount);

			glClear(GL_DEPTH_BUFFER_BIT);

			CVoxelGrid voxGrid512;
			VoxelDebugVAO vao512 = voxelScene512.VoxelDataForRendering(voxGrid512, debugVoxTransferTime, cache512.voxInfo.sceneVoxOctreeCount);
			DebugRenderVoxelPage(mainRenderCamera, vao512, voxGrid512, false,
								 cache512.voxInfo.sceneVoxOctreeCount);
			break;
		}
		case GI_VOXEL_CACHE512:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, cache512);
			break;
		}
		case GI_VOXEL_CACHE256:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, cache256);
			break;
		}
		case GI_VOXEL_CACHE128:
		{
			glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
			DebugRenderVoxelCache(mainRenderCamera, cache128);
			break;
		}
	}
}

void ThesisSolution::SetFPS(double fpsMS)
{
	frameTime = fpsMS;
}