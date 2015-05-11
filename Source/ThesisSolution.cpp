#include "ThesisSolution.h"
#include "Globals.h"
#include "SceneI.h"
#include "DrawBuffer.h"
#include "Macros.h"

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
}

void ThesisSolution::Frame(const Camera& mainWindowCamera)
{
	glClear(GL_COLOR_BUFFER_BIT);
	DrawBuffer& dBuffer = currentScene->getDrawBuffer();

	// Determine Voxel Sizes
	computeDetermineVoxSpan.Bind();

	objectGridInfo.Resize(currentScene->DrawCount());
	currentScene->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
	objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	glUniform1ui(U_TOTAL_OBJ_COUNT, static_cast<GLuint>(currentScene->DrawCount()));

	size_t blockCount = (currentScene->DrawCount() / VOXEL_SIZE);
	size_t factor = ((currentScene->DrawCount() % VOXEL_SIZE) == 0) ? 0 : 1;
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
	currentScene->getGPUBuffer().Bind();

	// Render Image
	voxelRenderTexture.BindAsImage(I_VOX_READ, GL_READ_WRITE);

	// State
	glEnable(GL_MULTISAMPLE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDepthMask(false);
	//glColorMask(false, false, false, false);

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

		// Material Buffer we need to fetch color from material
		dBuffer.BindMaterialForDraw(i);

		// We need to set viewport coords to match the voxel dims
		const AABBData& objAABB = currentScene->getDrawBuffer().getAABBBuffer().CPUData()[i];
		GLuint voxDimX, voxDimY, voxDimZ, one;
		one = 1;
		voxDimX = std::max(static_cast<GLuint>((objAABB.max.getX() - objAABB.min.getX()) / objectGridInfo.CPUData()[i].span), one);
		voxDimY = std::max(static_cast<GLuint>((objAABB.max.getY() - objAABB.min.getY()) / objectGridInfo.CPUData()[i].span), one);
		voxDimZ = std::max(static_cast<GLuint>((objAABB.max.getZ() - objAABB.min.getZ()) / objectGridInfo.CPUData()[i].span), one);

		glViewport(0, 0, voxDimX, voxDimY);
				   
		// Draw Call
		glDrawElementsIndirect(GL_TRIANGLES,
							   GL_UNSIGNED_INT,
							   (void *) (i * sizeof(DrawPointIndexed)));

		// Reflect Changes for the next process
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		// Second Call: Determine voxel count
		objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
		computeVoxelizeCount.Bind();
		size_t blockPerSliceX = (VOXEL_SIZE / voxDimX);
		size_t factor = ((VOXEL_SIZE % voxDimX) == 0) ? 0 : 1;
		blockPerSliceX += factor;
		size_t blockPerSliceY = (VOXEL_SIZE / voxDimY);
		factor = ((VOXEL_SIZE % voxDimY) == 0) ? 0 : 1;
		blockPerSliceY += factor;
		size_t blockPerSlice = blockPerSliceX + blockPerSliceY;
		glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
		glUniform4ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ, 
					 static_cast<GLuint>(blockPerSlice));

		// Compute Generation
		glDispatchCompute(static_cast<GLuint>(voxDimZ * blockPerSlice), 1, 1);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

		ObjGridInfo info = objectGridInfo.GetData(i);
		totalSceneVoxCount += info.voxCount;

		// Create sparse voxel array according to the size of voxel count

		// Last Call: Pack Draw Calls to the buffer

		// Voxelization Done!
	}

	GI_LOG("Total Vox : %d", totalSceneVoxCount);

	// DEBUG
	//objectGridInfo.SyncData(currentScene->DrawCount());
	//for(const ObjGridInfo& ogrd : objectGridInfo.CPUData())
	//{
	//	GI_LOG("\t%d", ogrd.voxCount);
	//}

	// Allocate Buffers According to the counts

	//// Same Thing bu render voxels to buffer
	//vertexVoxelizeObject.Bind();
	//fragmentVoxelizeObject.Bind();

	//glEnable(GL_MULTISAMPLE);
	//for(unsigned int i = 0; i < s.ObjectCount(); i++)
	//{
	//	const AABBData& objAABB = currentScene->getDrawBuffer().
	//		getAABBBuffer().
	//		CPUData()[i];
	//	cameraTransform.Update
	//		({
	//		IEMatrix4x4::Ortogonal(objAABB.min.getX(), objAABB.max.getX(),
	//		objAABB.min.getY(), objAABB.max.getY(),
	//		objAABB.min.getZ(), objAABB.max.getZ()),
	//		IEMatrix4x4::IdentityMatrix
	//	});

	//	// Render Object

	//}
//	glDisable(GL_MULTISAMPLE);

	// Render Voxel Cache of each object
}