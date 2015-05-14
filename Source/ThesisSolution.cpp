#include "ThesisSolution.h"
#include "Globals.h"
#include "SceneI.h"
#include "DrawBuffer.h"
#include "Macros.h"
#include "Camera.h"

size_t ThesisSolution::InitialObjectGridSize = 512;
size_t ThesisSolution::MaxVoxelCacheSize = 1024 * 1024 * 8;

ThesisSolution::ThesisSolution()
	: currentScene(nullptr)
	, vertexDebugVoxel(ShaderType::VERTEX, "Shaders/VoxRender.vert")
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
{
	voxelCacheUsageSize.AddData(0);
}

// Interface
bool ThesisSolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}

void ThesisSolution::Init(SceneI& s)
{
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	currentScene = &s;
	objectGridInfo.Resize(currentScene->DrawCount());

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

	size_t blockCount = (currentScene->DrawCount() / 128);
	size_t factor = ((currentScene->DrawCount() % 128) == 0) ? 0 : 1;
	blockCount += factor;
	glDispatchCompute(static_cast<GLuint>(blockCount), 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	objectGridInfo.SyncData(currentScene->DrawCount());

	// Render Objects to Voxel Grid
	// Use MSAA to prevent missing triangles on small voxels
	// (Instead of conservative rendering, visible surface determination)

	// Buffers
	cameraTransform.Bind();
	dBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();
	objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);

	// State
	glEnable(GL_MULTISAMPLE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDepthMask(false);
	//glColorMask(false, false, false, false);
	glViewport(0, 0, VOXEL_SIZE, VOXEL_SIZE);

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
	//	glMemoryBarrier(GL_ALL_BARRIER_BITS);

		// Second Call: Determine voxel count
		computeVoxelizeCount.Bind();
		voxelRenderTexture.BindAsImage(I_VOX_READ, GL_READ_ONLY);
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		glUniform3ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ);
		glDispatchCompute(VOXEL_SIZE / 8, VOXEL_SIZE / 8, VOXEL_SIZE / 8);

		// Reflect Changes to Next Process
	//	glMemoryBarrier(GL_ALL_BARRIER_BITS);

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
		glDispatchCompute(VOXEL_SIZE / 8, VOXEL_SIZE / 8, VOXEL_SIZE / 8);
//		glMemoryBarrier(GL_ALL_BARRIER_BITS);
		// Voxelization Done!
	}

	objectGridInfo.SyncData(currentScene->DrawCount());
	voxelCacheUsageSize.SyncData(1);
	uint32_t totalSceneVoxCount = 0;
	for(int i = 0; i < currentScene->DrawCount(); i++)
		totalSceneVoxCount += objectGridInfo.CPUData()[i].voxCount;
	GI_LOG("Total Vox : %d", totalSceneVoxCount);
	GI_LOG("Total Vox Written : %d", voxelCacheUsageSize.CPUData()[0]);
}

void ThesisSolution::Frame(const Camera& mainRenderCamera)
{
	// Frame Viewport
	glViewport(0, 0,
			   static_cast<GLsizei>(mainRenderCamera.width),
			   static_cast<GLsizei>(mainRenderCamera.height));

	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	glDisable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthMask(true);
	glColorMask(true, true, true, true);

	glClear(GL_COLOR_BUFFER_BIT |
			GL_DEPTH_BUFFER_BIT);

	// Debug Voxelize Scene
	Shader::Unbind(ShaderType::GEOMETRY);
	vertexDebugVoxel.Bind();
	fragmentDebugVoxel.Bind();

	cameraTransform.Bind();
	cameraTransform.Update(mainRenderCamera.generateTransform());

	objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	currentScene->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
	cameraTransform.Bind();
	
	voxelVAO.Bind();
	uint32_t offset = 0;
	for(unsigned int i = 0; i < currentScene->DrawCount(); i++)
	{
		//if((i == 33) ||
		//   (i == 34) ||
		//   (i == 5) ||
		//   (i == 4))
		{
			// Bind Model Transform
			DrawBuffer& dBuffer = currentScene->getDrawBuffer();
			dBuffer.getModelTransformBuffer().BindAsUniformBuffer(U_MTRANSFORM, i, 1);

			// Draw Call
			voxelVAO.Draw(objectGridInfo.CPUData()[i].voxCount, offset);
		}
		offset += objectGridInfo.CPUData()[i].voxCount;
	}
	
	//glClear(GL_COLOR_BUFFER_BIT);
	//DrawBuffer& dBuffer = currentScene->getDrawBuffer();

	//// Determine Voxel Sizes
	//computeDetermineVoxSpan.Bind();
	//objectGridInfo.Resize(currentScene->DrawCount());
	//currentScene->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
	//objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	//glUniform1ui(U_TOTAL_OBJ_COUNT, static_cast<GLuint>(currentScene->DrawCount()));

	//size_t blockCount = (currentScene->DrawCount() / 128);
	//size_t factor = ((currentScene->DrawCount() % 128) == 0) ? 0 : 1;
	//blockCount += factor;
	//glDispatchCompute(static_cast<GLuint>(blockCount), 1, 1);
	//glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	//objectGridInfo.SyncData(currentScene->DrawCount());

	//// Render Objects to Voxel Grid
	//// Use MSAA to prevent missing triangles on small voxels
	//// (Instead of conservative rendering, visible surface determination)

	//// Buffers
	//cameraTransform.Bind();
	//dBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();
	//objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);

	//// State
	//glEnable(GL_MULTISAMPLE);
	//glDisable(GL_DEPTH_TEST);
	//glDisable(GL_CULL_FACE);
	//glDepthMask(false);
	//glColorMask(false, false, false, false);
	//glViewport(0, 0, VOXEL_SIZE, VOXEL_SIZE);

	//// Reset Cache
	//voxelCacheUsageSize.CPUData()[0] = 0;
	//voxelCacheUsageSize.SendData();

	//// For Each Object
	//voxelRenderTexture.Clear();
	//for(unsigned int i = 0; i < currentScene->DrawCount(); i++)
	//{
	//	// First Call Voxelize over 3D Texture
	//	voxelRenderTexture.BindAsImage(I_VOX_WRITE, GL_WRITE_ONLY);
	//	vertexVoxelizeObject.Bind();
	//	glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
	//	geomVoxelizeObject.Bind();
	//	fragmentVoxelizeObject.Bind();
	//	glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
	//	currentScene->getGPUBuffer().Bind();

	//	// Material Buffer we need to fetch color from material
	//	dBuffer.BindMaterialForDraw(i);

	//	// We need to set viewport coords to match the voxel dims
	//	const AABBData& objAABB = currentScene->getDrawBuffer().getAABBBuffer().CPUData()[i];
	//	GLuint voxDimX, voxDimY, voxDimZ, one = 1;
	//	voxDimX = std::max(static_cast<GLuint>((objAABB.max.getX() - objAABB.min.getX()) / objectGridInfo.CPUData()[i].span), one);
	//	voxDimY = std::max(static_cast<GLuint>((objAABB.max.getY() - objAABB.min.getY()) / objectGridInfo.CPUData()[i].span), one);
	//	voxDimZ = std::max(static_cast<GLuint>((objAABB.max.getZ() - objAABB.min.getZ()) / objectGridInfo.CPUData()[i].span), one);

	//	// Draw Call
	//	glDrawElementsIndirect(GL_TRIANGLES,
	//						   GL_UNSIGNED_INT,
	//						   (void *) (i * sizeof(DrawPointIndexed)));

	//	// Reflect Changes for the next process
	//	glMemoryBarrier(GL_ALL_BARRIER_BITS);

	//	// Second Call: Determine voxel count
	//	computeVoxelizeCount.Bind();
	//	voxelRenderTexture.BindAsImage(I_VOX_READ, GL_READ_ONLY);
	//	glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
	//	glUniform3ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ);
	//	glDispatchCompute(VOXEL_SIZE / 8, VOXEL_SIZE / 8, VOXEL_SIZE / 8);

	//	// Reflect Changes to Next Process
	//	glMemoryBarrier(GL_ALL_BARRIER_BITS);

	//	// Create sparse voxel array according to the size of voxel count
	//	// Last Call: Pack Draw Calls to the buffer
	//	computePackObjectVoxels.Bind();
	//	voxelData.BindAsShaderStorageBuffer(LU_VOXEL);
	//	voxelRenderData.BindAsShaderStorageBuffer(LU_VOXEL_RENDER);
	//	voxelRenderTexture.BindAsImage(I_VOX_READ, GL_READ_WRITE);
	//	voxelCacheUsageSize.BindAsShaderStorageBuffer(LU_INDEX_CHECK);
	//	glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
	//	glUniform3ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ);
	//	glUniform1ui(U_MAX_CACHE_SIZE, static_cast<GLuint>(MaxVoxelCacheSize));
	//	glDispatchCompute(VOXEL_SIZE / 8, VOXEL_SIZE / 8, VOXEL_SIZE / 8);
	//	glMemoryBarrier(GL_ALL_BARRIER_BITS);
	//	// Voxelization Done!
	//}

	//objectGridInfo.SyncData(currentScene->DrawCount());
	//voxelCacheUsageSize.SyncData(1);
	//uint32_t totalSceneVoxCount = 0;
	//for(int i = 0; i < currentScene->DrawCount(); i++)
	//	totalSceneVoxCount += objectGridInfo.CPUData()[i].voxCount;
	//GI_LOG("Total Vox : %d", totalSceneVoxCount);
	//GI_LOG("Total Vox Written : %d", voxelCacheUsageSize.CPUData()[0]);
}