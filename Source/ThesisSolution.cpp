#include "ThesisSolution.h"

ThesisSolution::ThesisSolution()
	: currentScene(nullptr)
	, vertexDebugVoxel(ShaderType::VERTEX, "Shaders/GWriteGeneric.vert")
	, fragmentDebugVoxel(ShaderType::FRAGMENT, "Shaders/GWriteGeneric.frag")
	, vertexVoxelizeObject(ShaderType::VERTEX, "Shaders/VoxelizeGeom.vert")
	, fragmenVoxelizeObject(ShaderType::FRAGMENT, "Shaders/VoxelizeGeom.frag")
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