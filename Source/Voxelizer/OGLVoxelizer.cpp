#include "GLHeader.h"
#include <GLFW/glfw3.h>
#include "OGLVoxelizer.h"
#include "Macros.h"
#include "MeshBatch.h"
#include "GL3DTexture.h"
#include "Shader.h"
#include "BindPoints.h"

GLFWwindow* OGLVoxelizer::window = nullptr;

OGLVoxelizer::OGLVoxelizer(const VoxelizerOptions& options, 
						   MeshBatch& batch,
						   GL3DTexture& lockTex,
						   GL3DTexture& normalTex,
						   GL3DTexture& colorTex,
						   Shader& compSplitCount,
						   Shader& compPackVoxels,
						   Shader& vertVoxelize,
						   Shader& geomVoxelize,
						   Shader& fragVoxelize,
						   Shader& fragVoxelizeCount,
						   bool isSkeletal)
	: options(options)
	, batch(batch)
	, split(batch.DrawCount())
	, objVoxCount(batch.DrawCount())
	, totalVoxCount(1)
	, lockTex(lockTex)
	, normalTex(normalTex)
	, colorTex(colorTex)
	, compSplitCount(compSplitCount)
	, compPackVoxels(compPackVoxels)
	, vertVoxelize(vertVoxelize)
	, geomVoxelize(geomVoxelize)
	, fragVoxelize(fragVoxelize)
	, fragVoxelizeCount(fragVoxelizeCount)
	, isSkeletal(isSkeletal)
{
	assert(window != nullptr);
}

OGLVoxelizer::~OGLVoxelizer()
{

}

bool OGLVoxelizer::InitGLSystem()
{
	assert(window == nullptr);
	if(!glfwInit())
	{
		GI_ERROR_LOG("Fatal Error: Could not Init GLFW");
		return false;
	}
	glfwSetErrorCallback(ErrorCallbackGLFW);

	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
	glfwWindowHint(GLFW_SRGB_CAPABLE, GL_FALSE);	// Buggy

	glfwWindowHint(GLFW_RED_BITS, 8);
	glfwWindowHint(GLFW_GREEN_BITS, 8);
	glfwWindowHint(GLFW_BLUE_BITS, 8);
	glfwWindowHint(GLFW_ALPHA_BITS, 8);

	glfwWindowHint(GLFW_DEPTH_BITS, 24);
	glfwWindowHint(GLFW_STENCIL_BITS, 8);

	glfwWindowHint(GLFW_SAMPLES, 16);

	glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);

	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_NONE);

	#ifdef GI_DEBUG
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	#else
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_FALSE);
	#endif

	window = glfwCreateWindow(4, 4, "GI Thesis", nullptr, nullptr);
	if(window == nullptr)
	{
		GI_ERROR_LOG("Fatal Error: Could not create window.");
		return false;
	}

	glfwMakeContextCurrent(window);

	// Now Init GLEW
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if(err != GLEW_OK)
	{
		GI_ERROR_LOG("Error: %s\n", glewGetErrorString(err));
		assert(false);
	}

	// Print Stuff Now
	// Window Done
	GI_LOG("OGL Initialized.");
	GI_LOG("GLEW\t: %s", glewGetString(GLEW_VERSION));
	GI_LOG("GLFW\t: %s", glfwGetVersionString());
	GI_LOG("");
	GI_LOG("Renderer Information...");
	GI_LOG("OpenGL\t: %s", glGetString(GL_VERSION));
	GI_LOG("GLSL\t: %s", glGetString(GL_SHADING_LANGUAGE_VERSION));
	GI_LOG("Device\t: %s", glGetString(GL_RENDERER));
	GI_LOG("");

	#ifdef GI_DEBUG
		// Add Callback
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(OGLVoxelizer::OGLCallbackRender, nullptr);
		glDebugMessageControl(GL_DONT_CARE,
							  GL_DONT_CARE,
							  GL_DONT_CARE,
							  0,
							  nullptr,
							  GL_TRUE);
	#endif

	// Get Some GPU Limitations
	// DEBUG
	GLint uniformBufferOffsetAlignment, ssbOffsetAlignment;
	glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &uniformBufferOffsetAlignment);
	glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &ssbOffsetAlignment);

	glfwSwapInterval(0);
	return true;
}

void OGLVoxelizer::DestroyGLSystem()
{
	if(window)
	{
		glfwMakeContextCurrent(nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}
}

void OGLVoxelizer::ErrorCallbackGLFW(int error, const char* description)
{
	GI_ERROR_LOG("GLFW Error %d: %s", error, description);
}

void __stdcall OGLVoxelizer::OGLCallbackRender(GLenum,
											   GLenum type,
											   GLuint id,
											   GLenum severity,
											   GLsizei,
											   const GLchar* message,
											   const void*)
{
	// Dont Show Others For Now
	if(type == GL_DEBUG_TYPE_OTHER ||	//
	   id == 131186 ||					// Buffer Copy warning omit
	   id == 131218)					// Shader recompile cuz of state mismatch omit
	   return;

	GI_DEBUG_LOG("---------------------OGL-Callback-Render------------");
	GI_DEBUG_LOG("Message: %s", message);
	switch(type)
	{
		case GL_DEBUG_TYPE_ERROR:
			GI_DEBUG_LOG("Type: ERROR");
			break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			GI_DEBUG_LOG("Type: DEPRECATED_BEHAVIOR");
			break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			GI_DEBUG_LOG("Type: UNDEFINED_BEHAVIOR");
			break;
		case GL_DEBUG_TYPE_PORTABILITY:
			GI_DEBUG_LOG("Type: PORTABILITY");
			break;
		case GL_DEBUG_TYPE_PERFORMANCE:
			GI_DEBUG_LOG("Type: PERFORMANCE");
			break;
		case GL_DEBUG_TYPE_OTHER:
			GI_DEBUG_LOG("Type: OTHER");
			break;
	}

	GI_DEBUG_LOG("ID: %d", id);
	switch(severity)
	{
		case GL_DEBUG_SEVERITY_LOW:
			GI_DEBUG_LOG("Severity: LOW");
			break;
		case GL_DEBUG_SEVERITY_MEDIUM:
			GI_DEBUG_LOG("Severity: MEDIUM");
			break;
		case GL_DEBUG_SEVERITY_HIGH:
			GI_DEBUG_LOG("Severity: HIGH");
			break;
		default:
			GI_DEBUG_LOG("Severity: NONE");
			break;
	}
	GI_DEBUG_LOG("---------------------OGL-Callback-Render-End--------------");
}

void OGLVoxelizer::DetermineSplits()
{
	auto& aabbBuffer = batch.getDrawBuffer().getAABBBuffer();

	// Determine Render Count for Each Object
	compSplitCount.Bind();

	// Uniforms Constants
	glUniform1ui(U_TOTAL_OBJ_COUNT, static_cast<GLuint>(batch.DrawCount()));
	glUniform1f(U_SPAN, options.span);
	glUniform1ui(U_GRID_DIM, VOX_3D_TEX_SIZE);

	// ShaderStorage
	aabbBuffer.BindAsShaderStorageBuffer(LU_AABB);
	split.BindAsShaderStorageBuffer(LU_OBJECT_SPLIT_INFO);

	size_t blockCount = (batch.DrawCount() + BLOCK_SIZE - 1) / BLOCK_SIZE;
	glDispatchCompute(static_cast<GLuint>(blockCount), 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	split.RecieveData(batch.DrawCount());
}

void OGLVoxelizer::AllocateVoxelCaches()
{
	// Shaders Common
	geomVoxelize.Bind();

	// Buffers
	auto& drawBuffer = batch.getDrawBuffer().getAABBBuffer();
	auto& aabbBuffer = batch.getDrawBuffer().getAABBBuffer();
	auto& gpuBuffer = batch.getGPUBuffer();

	gpuBuffer.Bind();
	drawBuffer.BindAsDrawIndirectBuffer();
	aabbBuffer.BindAsShaderStorageBuffer(LU_AABB);
	totalVoxCount.BindAsShaderStorageBuffer(LU_TOTAL_VOX_COUNT);
	objVoxCount.BindAsShaderStorageBuffer(LU_OBJECT_VOXEL_INFO);
	
	objVoxCount.Memset(0x00);
	totalVoxCount.Memset(0x00);

	// Images
	lockTex.BindAsImage(I_LOCK, GL_READ_WRITE);

	// States
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnable(GL_MULTISAMPLE);

	glDepthMask(false);
	glStencilMask(0x0000);
	glColorMask(false, false, false, false);

	glViewport(0, 0, VOX_3D_TEX_SIZE, VOX_3D_TEX_SIZE);
	
	// For Each Object
	for(GLuint i = 0; i < batch.DrawCount(); i++)
	{
		// Vertex Shader
		vertVoxelize.Bind();
		glUniform1ui(U_OBJ_ID, i);

		// Fragment Shader
		fragVoxelizeCount.Bind();
		glUniform1ui(U_OBJ_ID, i);
		glUniform1f(U_SPAN, options.span);

		// Split Geom to segments
		auto& voxSplit = split.CPUData()[i].voxSplit;
		for(uint32_t a = 0; a < voxSplit[0]; a++)
		for(uint32_t b = 0; b < voxSplit[1]; b++)
		for(uint32_t c = 0; c < voxSplit[2]; c++)
		{
			// 
			
		}
	}


}

float OGLVoxelizer::Voxelize()
{
	// Timing Voxelization Process
	GLuint queryID;
	glGenQueries(1, &queryID);
	glBeginQuery(GL_TIME_ELAPSED, queryID);

	// Split Determination
	DetermineSplits();

	// Voxel Count Determination
	AllocateVoxelCaches();
	
	//// Buffers
	//cameraTransform.Bind();
	//dBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();
	//cache.objectGridInfo.BindAsShaderStorageBuffer(LU_OBJECT_GRID_INFO);

	//// Render Objects to Voxel Grid
	//// Use MSAA to prevent missing small triangles on voxels
	//// (testing conservative rendering on maxwell)
	//glEnable(GL_MULTISAMPLE);
	////glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);

	//// State
	//glDisable(GL_DEPTH_TEST);
	//glDisable(GL_CULL_FACE);
	//glDepthMask(false);
	//glStencilMask(0x0000);
	//glColorMask(false, false, false, false);
	//glViewport(0, 0, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE);

	//// Reset Cache
	//cache.voxelCacheUsageSize.CPUData()[0] = 0;
	//cache.voxelCacheUsageSize.SendData();

	//// isMip
	//std::vector<GLuint> isMip(batch->DrawCount(), 0);
	//for(unsigned int i = 0; i < isMip.size(); i++)
	//{
	//	if(cache.objectGridInfo.CPUData()[i].span < batch->MinSpan() * minSpanMultiplier)
	//	{
	//		isMip[i] = (isInnerCascade) ? 0 : 1;
	//		cache.objectGridInfo.CPUData()[i].span = batch->MinSpan() * minSpanMultiplier;
	//	}

	//}
	//cache.objectGridInfo.SendData();

	//// For Each Object
	//voxelRenderTexture.Clear();
	//for(unsigned int i = 0; i < batch->DrawCount(); i++)
	//{
	//	// Skip objects that cant fit
	//	if(cache.objectGridInfo.CPUData()[i].span != batch->MinSpan() * minSpanMultiplier)
	//		continue;

	//	// First Call Voxelize over 3D Texture
	//	batch->getDrawBuffer().getAABBBuffer().BindAsShaderStorageBuffer(LU_AABB);
	//	voxelRenderTexture.BindAsImage(I_VOX_WRITE, GL_WRITE_ONLY);
	//	vertexVoxelizeObject.Bind();
	//	glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
	//	geomVoxelizeObject.Bind();
	//	fragmentVoxelizeObject.Bind();
	//	glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
	//	batch->getGPUBuffer().Bind();

	//	// Material Buffer we need to fetch color from material
	//	dBuffer.BindMaterialForDraw(i);

	//	// Draw Call
	//	glDrawElementsIndirect(GL_TRIANGLES,
	//						   GL_UNSIGNED_INT,
	//						   (void *)(i * sizeof(DrawPointIndexed)));

	//	// Reflect Changes for the next process
	//	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	//	// Second Call: Determine voxel count
	//	// We need to set viewport coords to match the voxel dims
	//	const AABBData& objAABB = batch->getDrawBuffer().getAABBBuffer().CPUData()[i];
	//	GLuint voxDimX, voxDimY, voxDimZ;
	//	voxDimX = static_cast<GLuint>(std::floor((objAABB.max.getX() - objAABB.min.getX()) / cache.objectGridInfo.CPUData()[i].span)) + 1;
	//	voxDimY = static_cast<GLuint>(std::floor((objAABB.max.getY() - objAABB.min.getY()) / cache.objectGridInfo.CPUData()[i].span)) + 1;
	//	voxDimZ = static_cast<GLuint>(std::floor((objAABB.max.getZ() - objAABB.min.getZ()) / cache.objectGridInfo.CPUData()[i].span)) + 1;

	//	computeVoxelizeCount.Bind();
	//	voxelRenderTexture.BindAsImage(I_VOX_READ, GL_READ_ONLY);
	//	glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
	//	glUniform3ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ);
	//	glDispatchCompute(voxDimX + 7 / 8, voxDimY + 7 / 8, voxDimZ + 7 / 8);

	//	// Reflect Voxel Size
	//	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	//	// Create sparse voxel array according to the size of voxel count
	//	// Last Call: Pack Draw Calls to the buffer
	//	computePackObjectVoxels.Bind();
	//	cache.voxelNormPos.BindAsShaderStorageBuffer(LU_VOXEL_NORM_POS);
	//	cache.voxelIds.BindAsShaderStorageBuffer(LU_VOXEL_IDS);
	//	cache.voxelRenderData.BindAsShaderStorageBuffer(LU_VOXEL_RENDER);
	//	cache.voxelCacheUsageSize.BindAsShaderStorageBuffer(LU_INDEX_CHECK);
	//	voxelRenderTexture.BindAsImage(I_VOX_READ, GL_READ_WRITE);
	//	glUniform1ui(U_OBJ_TYPE, static_cast<GLuint>(batch->MeshType()));
	//	glUniform1ui(U_OBJ_ID, static_cast<GLuint>(i));
	//	glUniform3ui(U_TOTAL_VOX_DIM, voxDimX, voxDimY, voxDimZ);
	//	glUniform1ui(U_MAX_CACHE_SIZE, static_cast<GLuint>(cache.voxelNormPos.Capacity()));
	//	glUniform1ui(U_IS_MIP, static_cast<GLuint>(isMip[i]));
	//	glDispatchCompute(voxDimX + 7 / 8, voxDimY + 7 / 8, voxDimZ + 7 / 8);
	//	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	//	// Voxelization Done!
	//}
	////glDisable(GL_CONSERVATIVE_RASTERIZATION_NV);
	glEndQuery(GL_TIME_ELAPSED);
	//glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	//cache.objectGridInfo.RecieveData(batch->DrawCount());
	//cache.voxelCacheUsageSize.RecieveData(1);
	//cache.batchVoxCacheCount = 0;
	//for(int i = 0; i < batch->DrawCount(); i++)
	//	cache.batchVoxCacheCount += cache.objectGridInfo.CPUData()[i].voxCount;
	//assert(cache.voxelCacheUsageSize.CPUData()[0] == cache.batchVoxCacheCount);

	// Check if we exceeded the max (normally we didnt write bu we incremented counter)
	//cache.batchVoxCacheCount = std::min(cache.batchVoxCacheCount,
	//									static_cast<uint32_t>(cache.voxelNormPos.Capacity()));
	//cache.batchVoxCacheSize = static_cast<double>(cache.batchVoxCacheCount *
	//											  (sizeof(CVoxelNormPos) +
	//											  sizeof(VoxelColorData) +
	//											  sizeof(CVoxelIds))) /
	//											  1024.0 /
	//											  1024.0;

	// Timing
	GLuint64 timeElapsed = 0;
	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);
	glDeleteQueries(1, &queryID);
	double time = timeElapsed / 1000000.0;

	GI_LOG("Voxelization Complete");
	GI_LOG("Scene Voxelization Time: %f ms", time);
	GI_LOG("Total Batch Vox : %d", 123);
	//GI_LOG("Total Vox Memory: %f MB", cache.batchVoxCacheSize);
	GI_LOG("------------------------------------");

	
	return static_cast<float>(time);
	//return 0.0;
}