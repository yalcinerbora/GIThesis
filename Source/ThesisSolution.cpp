#include "ThesisSolution.h"
#include "Globals.h"
#include "SceneI.h"
#include "DrawBuffer.h"

ThesisSolution::ThesisSolution()
	: currentScene(nullptr)
	, vertexDebugVoxel(ShaderType::VERTEX, "Shaders/VoxRender.vert")
	, fragmentDebugVoxel(ShaderType::FRAGMENT, "Shaders/VoxRender.frag")
	, vertexVoxelizeObject(ShaderType::VERTEX, "Shaders/VoxelizeGeom.vert")
	, fragmentVoxelizeObject(ShaderType::FRAGMENT, "Shaders/VoxelizeGeom.frag")
	, fragmentVoxelizeCount(ShaderType::FRAGMENT, "Shaders/VoxelizeGeomCount.frag")
	, computeDetermineVoxSpan(ShaderType::COMPUTE, "Shaders/DetermineVoxSpan.glsl")
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

	// Count Determination
	glGenBuffers(1, &objGridInfoBuffer);
	glBindBuffer(GL_COPY_WRITE_BUFFER, objGridInfoBuffer);
	glBufferData(GL_COPY_WRITE_BUFFER,
				 currentScene->ObjectCount() * (sizeof(float) + sizeof(unsigned int)),
				 nullptr,
				 GL_DYNAMIC_DRAW);

	glClearBufferData(GL_COPY_WRITE_BUFFER,
					  GL_R32UI,
					  GL_R,
					  GL_UNSIGNED_INT,
					  nullptr);

	//vertexVoxelizeObject.Bind();
	//fragmentVoxelizeCount.Bind();


	// Bind Related Data
	// aabbBufferId = currentScene->getDrawBuffer().GetAABBBuffer();
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, LU_OBJECT, aabbBufferId);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, LU_OBJECT_GRID_INFO, objGridInfoBuffer);
	//glUniform1i(U_TOTAL_OBJ_COUNT, static_cast<GLint>(currentScene->ObjectCount()));


	// Launch Compute
	GLuint blockCount = currentScene->ObjectCount() / 128 + 
							(currentScene->ObjectCount() % 128 == 0) ? 0 : 1;
	glDispatchComputeGroupSizeARB(128, 1, 1, blockCount,
								  1, 1);


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