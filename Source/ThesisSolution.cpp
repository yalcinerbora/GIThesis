#include "ThesisSolution.h"
#include "Globals.h"
#include "SceneI.h"
#include "DrawBuffer.h"
#include "Macros.h"
#include "Camera.h"

size_t ThesisSolution::InitialObjectGridSize = 512;

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
{}

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

	// Render Image
	voxelRenderTexture.BindAsImage(I_VOX_READ, GL_READ_WRITE);

	// State
	glEnable(GL_MULTISAMPLE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDepthMask(false);
	glColorMask(false, false, false, false);
	glViewport(0, 0, VOXEL_SIZE, VOXEL_SIZE);

	// For Each Object
	uint32_t totalSceneVoxCount = 0;
	for(unsigned int i = 0; i < currentScene->DrawCount(); i++)
	{
		// Clear Voxel 3D Texture
		voxelRenderTexture.Clear();

		// First Call Voxelize over 3D Texture
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
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		// Second Call: Determine voxel count
		computeVoxelizeCount.Bind();
		size_t blockPerSliceX = (voxDimX / 32);
		size_t factor = ((voxDimX % 32) == 0) ? 0 : 1;
		blockPerSliceX += factor;
		size_t blockPerSliceY = (voxDimY / 32);
		factor = ((voxDimY % 32) == 0) ? 0 : 1;
		blockPerSliceY += factor;
		size_t blockPerSlice = blockPerSliceX * blockPerSliceY;
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		glUniform3ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ);
		glUniform2ui(U_VOX_SLICE,
					 static_cast<GLuint>(blockPerSliceX),
					 static_cast<GLuint>(blockPerSlice));
		glDispatchCompute(static_cast<GLuint>(voxDimZ * blockPerSlice), 1, 1);

		// Reflect Changes to Next Process
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | 
						GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		ObjGridInfo info = objectGridInfo.GetData(i);
		totalSceneVoxCount += info.voxCount;

		// Create sparse voxel array according to the size of voxel count
		voxelData.emplace_back(info.voxCount);
		voxelRenderData.emplace_back(info.voxCount);

		// Last Call: Pack Draw Calls to the buffer
		computePackObjectVoxels.Bind();
		voxelData.back().BindAsShaderStorageBuffer(LU_VOXEL);
		voxelRenderData.back().BindAsShaderStorageBuffer(LU_VOXEL_RENDER);
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		glUniform3ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ);
		glUniform2ui(U_VOX_SLICE,
					 static_cast<GLuint>(blockPerSliceX),
					 static_cast<GLuint>(blockPerSlice));
		glDispatchCompute(static_cast<GLuint>(voxDimZ * blockPerSlice), 1, 1);
		glMemoryBarrier(GL_ALL_BARRIER_BITS);

		// Debug VAO
		voxelVAO.emplace_back(voxelData.back(), voxelRenderData.back(), info.voxCount);

		// Voxelization Done!
	}
	GI_LOG("Total Vox : %d", totalSceneVoxCount);

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
	
	auto voxelVaoIt = voxelVAO.begin();
	for(unsigned int i = 0; i < 100; i++)
	{
		// Bind Model Transform
		DrawBuffer& dBuffer = currentScene->getDrawBuffer();
		dBuffer.getModelTransformBuffer().BindAsUniformBuffer(U_MTRANSFORM, i, 1);

		// Draw Call
		voxelVaoIt->Draw();
		voxelVaoIt++;
	}
}