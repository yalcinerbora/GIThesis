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

	objectGridInfo.Resize(currentScene->DrawCount());
}

void ThesisSolution::Frame(const Camera& mainWindowCamera)
{
	glClear(GL_COLOR_BUFFER_BIT);
	DrawBuffer& dBuffer = currentScene->getDrawBuffer();

	// Determine Voxel Sizes
	computeDetermineVoxSpan.Bind();

	objectGridInfo.Resize(currentScene->DrawCount());
	currentScene->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_OBJECT);
	objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	glUniform1ui(U_TOTAL_OBJ_COUNT, static_cast<GLuint>(currentScene->DrawCount()));

	// Launch Compute
	size_t blockCount = (currentScene->DrawCount() / 128);
	size_t factor = ((currentScene->DrawCount() % 128) == 0) ? 0 : 1;
	blockCount += factor;
	glDispatchCompute(static_cast<GLuint>(blockCount), 1, 1);
	glMemoryBarrier(GL_ALL_BARRIER_BITS);
	objectGridInfo.SyncData(currentScene->DrawCount());

	// Determine each ojects total voxel count (sparse)
	vertexVoxelizeObject.Bind();
	fragmentVoxelizeCount.Bind();

	cameraTransform.Bind();
	dBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();
	objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);
	currentScene->getGPUBuffer().Bind();

	glEnable(GL_MULTISAMPLE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDepthMask(false);
	//glColorMask(false, false, false, false);
	for(unsigned int i = 0; i < currentScene->DrawCount(); i++)
	{
		const AABBData& objAABB = currentScene->getDrawBuffer().
			getAABBBuffer().
			CPUData()[i];
		glUniform1ui(U_TOTAL_OBJ_COUNT, static_cast<GLuint>(i));


		//DEBUG
		dBuffer.BindMaterialForDraw(i);
		if(i >= 324 &&
		   i <= 330)
		GI_LOG("%d\tx\t%d", 
			   static_cast<GLsizei>((objAABB.max.getX() - objAABB.min.getX()) / objectGridInfo.CPUData()[i].span),
			   static_cast<GLsizei>((objAABB.max.getY() - objAABB.min.getY()) / objectGridInfo.CPUData()[i].span));

		glViewport(0, 0,
				   std::max(static_cast<GLsizei>((objAABB.max.getX() - objAABB.min.getX()) / objectGridInfo.CPUData()[i].span), 1),
				   std::max(static_cast<GLsizei>((objAABB.max.getY() - objAABB.min.getY()) / objectGridInfo.CPUData()[i].span), 1));
		cameraTransform.Update
		({
			IEMatrix4x4::IdentityMatrix,
			IEMatrix4x4::Ortogonal(objAABB.min.getX(), objAABB.max.getX(),
									objAABB.max.getY(), objAABB.min.getY(),
									objAABB.min.getZ(), objAABB.max.getZ()),
			IEMatrix4x4::IdentityMatrix
		});

		glDrawElementsIndirect(GL_TRIANGLES,
							   GL_UNSIGNED_INT,
							   (void *) (i * sizeof(DrawPointIndexed)));
	}
	glMemoryBarrier(GL_ALL_BARRIER_BITS);

	GI_LOG("----------------------------------------------");

	// DEBUG
	objectGridInfo.SyncData(currentScene->DrawCount());
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