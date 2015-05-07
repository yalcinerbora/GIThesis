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
	, fragmentVoxelizeObject(ShaderType::FRAGMENT, "Shaders/VoxelizeGeom.frag")
	, fragmentVoxelizeCount(ShaderType::FRAGMENT, "Shaders/VoxelizeGeomCount.frag")
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

	// Determine Voxel Sizes
	computeDetermineVoxSpan.Bind();

	objectGridInfo.Resize(s.ObjectCount());
	currentScene->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_OBJECT);
	objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	glUniform1ui(U_TOTAL_OBJ_COUNT, static_cast<GLint>(currentScene->ObjectCount()));

	// Launch Compute
	size_t blockCount = (currentScene->ObjectCount() / 128);
	size_t factor = ((currentScene->ObjectCount() % 128) == 0) ? 0 : 1;
	blockCount += factor;
	//glDispatchComputeGroupSizeARB(static_cast<GLuint>(blockCount), 1, 1,
	//							  128, 1, 1);
	glDispatchCompute(static_cast<GLuint>(blockCount), 1, 1);

	glMemoryBarrier(GL_ALL_BARRIER_BITS);

	// DEBUG
	objectGridInfo.SyncData(currentScene->ObjectCount());
	/*for(const AABBData& aabb : currentScene->getDrawBuffer().getAABBBuffer().CPUData())
	{
		IEVector4 dif = aabb.max - aabb.min;
		GI_LOG("\t%f, %f, %f", dif.getX(), dif.getY(), dif.getZ());
	}*/
	for(const ObjGridInfo& ogrd : objectGridInfo.CPUData())
	{
		GI_LOG("\t%f", ogrd.span);
	}

	
	//// Determine each objects total voxel count (sparse)
	//vertexVoxelizeObject.Bind();
	//fragmentVoxelizeCount.Bind();

	//for(unsigned int i = 0; i < s.ObjectCount(); i++)
	//{
	//	const AABBData& objAABB = currentScene->getDrawBuffer().getAABBBuffer().CPUImage()[i];

	//	cameraTransform.Update
	//	({
	//		IEMatrix4x4::Ortogonal(objAABB.min.getX(), objAABB.max.getX(),
	//								objAABB.min.getY(), objAABB.max.getY(),
	//								objAABB.min.getZ(), objAABB.max.getZ()),
	//		IEMatrix4x4::IdentityMatrix
	//	});
	//	
	//	// Render Objects


	//}

	
	// Get the scene
	// Create every objects voxel data relative to their span
		// Run Voxel Render Shader Twice
		// First to get data size for allocation
		// Second to actually writing the data

	// Store per object data



}

void ThesisSolution::Frame(const Camera& mainWindowCamera)
{
	glClear(GL_COLOR_BUFFER_BIT);


	// Render Voxel Cache of each object
}