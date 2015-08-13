#include "ThesisSolution.h"
#include "Globals.h"
#include "SceneI.h"
#include "DrawBuffer.h"
#include "Macros.h"
#include "Camera.h"
#include "DeferredRenderer.h"
#include "SceneLights.h"

size_t ThesisSolution::InitialObjectGridSize = 256;
size_t ThesisSolution::MaxVoxelCacheSize = 1024 * 1024 * 8;

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
	, objectGridInfo(InitialObjectGridSize)
	, voxelData(MaxVoxelCacheSize)
	, voxelRenderData(MaxVoxelCacheSize)
	, voxelCacheUsageSize(1)
	, voxelVAO(voxelData,voxelRenderData)
	, voxInfo({0})
	, bar(nullptr)
	, relativeTransformBuffer(1)
	, voxelScene(CVoxelGrid{{intialCamPos.getX() - (512.0f * 0.5f * 0.5f), intialCamPos.getY() - (512.0f * 0.5f * 0.5f), intialCamPos.getZ() - (512.0f * 0.5f *  0.5f)},
							0.5f, {256, 256, 256}, 9})
{
	voxelCacheUsageSize.AddData(0);
	voxelScene.LinkDeferredRendererBuffers(dRenderer.GetGBuffer().getDepthGLView(),
										   dRenderer.GetGBuffer().getNormalGL(),
										   dRenderer.GetLightIntensityBufferGL());
}

ThesisSolution::~ThesisSolution()
{
	voxelScene.UnLinkDeferredRendererBuffers();
}

bool ThesisSolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}

void ThesisSolution::Init(SceneI& s)
{
	// Reset GICudaScene
	voxelScene.Reset();

	// Initialiizng Relative Transform Buffer
	for(unsigned int i = 0; i < s.ObjectCount(); i++)
	{
		relativeTransformBuffer.AddData(ModelTransform { IEMatrix4x4::IdentityMatrix, IEMatrix4x4::IdentityMatrix });
	}
	relativeTransformBuffer.SendData();
	
	// Voxelization
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	currentScene = &s;
	objectGridInfo.Resize(currentScene->DrawCount());

	GLuint queryID;
	glGenQueries(1, &queryID);
	glBeginQuery(GL_TIME_ELAPSED, queryID);

	//
	glClear(GL_COLOR_BUFFER_BIT);
	DrawBuffer& dBuffer = currentScene->getDrawBuffer();
	VoxelRenderTexture voxelRenderTexture;

	// Determine Voxel Sizes
	computeDetermineVoxSpan.Bind();
	objectGridInfo.Resize(currentScene->DrawCount());
	currentScene->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
	objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	glUniform1ui(U_TOTAL_OBJ_COUNT, static_cast<GLuint>(currentScene->DrawCount()));
	glUniform1f(U_MIN_SPAN, currentScene->MinSpan());

	size_t blockCount = (currentScene->DrawCount() / 128);
	size_t factor = ((currentScene->DrawCount() % 128) == 0) ? 0 : 1;
	blockCount += factor;
	glDispatchCompute(static_cast<GLuint>(blockCount), 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	objectGridInfo.RecieveData(currentScene->DrawCount());

	// Render Objects to Voxel Grid
	// Use MSAA to prevent missing small triangles on voxels
	// (Instead of conservative rendering)

	// Buffers
	cameraTransform.Bind();
	dBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();
	objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);

	// State
	glEnable(GL_MULTISAMPLE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDepthMask(false);
	glStencilMask(0x0000);
	glColorMask(false, false, false, false);
	glViewport(0, 0, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE);

	// Reset Cache
	voxelCacheUsageSize.CPUData()[0] = 0;
	voxelCacheUsageSize.SendData();

	// For Each Object
	voxelRenderTexture.Clear();

	for(unsigned int i = 0; i < currentScene->DrawCount(); i++)
	{
		// First Call Voxelize over 3D Texture
		voxelRenderTexture.BindAsImage(I_VOX_WRITE, GL_WRITE_ONLY);
		vertexVoxelizeObject.Bind();
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		geomVoxelizeObject.Bind();
		fragmentVoxelizeObject.Bind();
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		currentScene->getGPUBuffer().Bind();

		// Material Buffer we need to fetch color from material
		dBuffer.BindMaterialForDraw(i);

		// We need to set viewport coords to match the voxel dims
		const AABBData& objAABB = currentScene->getDrawBuffer().getAABBBuffer().CPUData()[i];
		GLuint voxDimX, voxDimY, voxDimZ, one = 1;
		voxDimX = std::max(static_cast<GLuint>((objAABB.max.getX() - objAABB.min.getX()) / objectGridInfo.CPUData()[i].span), one);
		voxDimY = std::max(static_cast<GLuint>((objAABB.max.getY() - objAABB.min.getY()) / objectGridInfo.CPUData()[i].span), one);
		voxDimZ = std::max(static_cast<GLuint>((objAABB.max.getZ() - objAABB.min.getZ()) / objectGridInfo.CPUData()[i].span), one);

		// Draw Call
		glDrawElementsIndirect(GL_TRIANGLES,
							   GL_UNSIGNED_INT,
							   (void *) (i * sizeof(DrawPointIndexed)));

		
		// Reflect Changes for the next process
//		glMemoryBarrier(GL_ALL_BARRIER_BITS);

		// Second Call: Determine voxel count
		computeVoxelizeCount.Bind();
		voxelRenderTexture.BindAsImage(I_VOX_READ, GL_READ_ONLY);
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		glUniform3ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ);
		glDispatchCompute(VOXEL_GRID_SIZE / 8, VOXEL_GRID_SIZE / 8, VOXEL_GRID_SIZE / 8);

		// Reflect Changes to Next Process
//		glMemoryBarrier(GL_ALL_BARRIER_BITS);

		// Create sparse voxel array according to the size of voxel count
		// Last Call: Pack Draw Calls to the buffer
		computePackObjectVoxels.Bind();
		voxelData.BindAsShaderStorageBuffer(LU_VOXEL);
		voxelRenderData.BindAsShaderStorageBuffer(LU_VOXEL_RENDER);
		voxelRenderTexture.BindAsImage(I_VOX_READ, GL_READ_WRITE);
		voxelCacheUsageSize.BindAsShaderStorageBuffer(LU_INDEX_CHECK);
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		glUniform3ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ);
		glUniform1ui(U_MAX_CACHE_SIZE, static_cast<GLuint>(MaxVoxelCacheSize));
		glDispatchCompute(VOXEL_GRID_SIZE / 8, VOXEL_GRID_SIZE / 8, VOXEL_GRID_SIZE / 8);
//		glMemoryBarrier(GL_ALL_BARRIER_BITS);
		// Voxelization Done!
	}
	glEndQuery(GL_TIME_ELAPSED);

	GLuint64 timeElapsed;
	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);

	objectGridInfo.RecieveData(currentScene->DrawCount());
	voxelCacheUsageSize.RecieveData(1);
	voxInfo.sceneVoxCacheCount = 0;
	for(int i = 0; i < currentScene->DrawCount(); i++)
		voxInfo.sceneVoxCacheCount += objectGridInfo.CPUData()[i].voxCount;
	voxInfo.sceneVoxCacheSize = static_cast<double>(voxInfo.sceneVoxCacheCount * (sizeof(VoxelData) + sizeof(VoxelRenderData))) / (1024.0 * 1024.0);

	GI_LOG("------------------------------------");
	GI_LOG("GI Thesis Solution Init Complete");
	GI_LOG("Scene Voxelization Time: %f ms", timeElapsed / 1000000.0);
	GI_LOG("Total Vox : %d", voxInfo.sceneVoxCacheCount);
	GI_LOG("Total Vox Memory: %f MB", voxInfo.sceneVoxCacheSize);
	GI_LOG("------------------------------------");

	glDeleteQueries(1, &queryID);

	// Tw Bar Creation
	bar = TwNewBar("ThesisGI");
	TwDefine(" ThesisGI refresh=0.01 ");


	// Send it to CUDA
	voxelScene.LinkOGL(currentScene->getDrawBuffer().getAABBBuffer().getGLBuffer(),
					   currentScene->getDrawBuffer().getModelTransformBuffer().getGLBuffer(),
					   relativeTransformBuffer.getGLBuffer(),
					   objectGridInfo.getGLBuffer(),
					   voxelData.getGLBuffer(),
					   voxelRenderData.getGLBuffer(),
					   static_cast<uint32_t>(currentScene->ObjectCount()),
					   voxInfo.sceneVoxCacheCount);
	// Link ShadowMaps and GBuffer textures to cuda
	voxelScene.LinkSceneTextures(currentScene->getSceneLights().GetShadowMapArrayR32F());
	voxelScene.AllocateInitialPages(static_cast<uint32_t>(voxInfo.sceneVoxCacheCount * 4.0f));
	
	// FPS Show
	TwAddVarRO(bar, "fTime", TW_TYPE_DOUBLE, &frameTime,
			   " label='Frame(ms)' precision=2 help='Frame Time in milliseconds.' ");
	TwAddSeparator(bar, NULL, NULL);
	TwAddVarRO(bar, "voxCache", TW_TYPE_UINT32, &voxInfo.sceneVoxCacheCount,
			   " label='Voxel Count' group='Voxel Cache' help='Cache voxel count.' ");
	TwAddVarRO(bar, "voxCacheSize", TW_TYPE_DOUBLE, &voxInfo.sceneVoxCacheSize,
			   " label='Voxel Size(MB)' group='Voxel Cache' precision=2 help='Cache voxel total size in megabytes.' ");
	TwAddSeparator(bar, NULL, NULL);
	TwAddVarRO(bar, "voxUsed", TW_TYPE_UINT32, &voxInfo.sceneVoxOctreeCount,
			   " label='Voxel Used' group='Voxel Octree' help='Voxel count in octree.' ");
	TwAddVarRO(bar, "voxUsedSize", TW_TYPE_DOUBLE, &voxInfo.sceneVoxOctreeSize,
			   " label='Voxel Used Size(MB)' group='Voxel Octree' precision=2 help='Octree Voxel total size in megabytes.' ");
	TwAddSeparator(bar, NULL, NULL);
	TwAddVarRO(bar, "ioTime", TW_TYPE_DOUBLE, &ioTime,
			   " label='I-O Time (ms)' group='Timings' precision=2 help='Voxel Include Exclude Timing per frame.' ");
	TwAddVarRO(bar, "updateTime", TW_TYPE_DOUBLE, &transformTime,
			   " label='Update Time (ms)' group='Timings' precision=2 help='Voxel Grid Update Timing per frame.' ");
	TwAddVarRO(bar, "svoReconTime", TW_TYPE_DOUBLE, &svoTime,
			   " label='SVO Time (ms)' group='Timings' precision=2 help='SVO Reconstruct Timing per frame.' ");
}

void ThesisSolution::Release()
{
	if(bar) TwDeleteBar(bar);
}

void ThesisSolution::DebugRenderVoxelCache(const Camera& camera)
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

	objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	currentScene->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
	cameraTransform.Bind();

	// Bind Model Transform
	DrawBuffer& dBuffer = currentScene->getDrawBuffer();
	dBuffer.getModelTransformBuffer().BindAsShaderStorageBuffer(LU_MTRANSFORM);

	voxelVAO.Bind();
	voxelVAO.Draw(voxInfo.sceneVoxCacheCount, 0);
}

void ThesisSolution::Frame(const Camera& mainRenderCamera)
{
	// DEBUG
	DebugRenderVoxelCache(mainRenderCamera);

	// VoxelSceneUpdate
	voxelScene.Voxelize(ioTime,
						transformTime,
						svoTime,
						mainRenderCamera.pos);

	// Voxel Count in Pages
	voxInfo.sceneVoxOctreeCount = voxelScene.VoxelCountInPage();
	voxInfo.sceneVoxOctreeSize = static_cast<double>(voxInfo.sceneVoxOctreeCount * sizeof(uint32_t) * 4) / 1024 / 1024;
	
	// Here check TW Bar if user wants to render voxels


	// Paged voxel rendering
	//uint32_t voxCount = 0;
	//VoxelDebugVAO& voxelVao = voxelScene.VoxelDataForRendering(voxCount);
	//voxelVao.Bind();
	//voxelVAO.Draw(voxCount, 0);


	
	// Or wants to render only voxel light contrubition to the scene

	// Or renders the scene as whole
	//dRenderer.Render(*currentScene, mainRenderCamera);
}

void ThesisSolution::SetFPS(double fpsMS)
{
	frameTime = fpsMS;
}