#include "GLHeader.h"
#include <GLFW/glfw3.h>
#include "OGLVoxelizer.h"
#include "Macros.h"
#include "MeshBatch.h"
#include "GL3DTexture.h"
#include "Shader.h"
#include "BindPoints.h"
#include "OGLTimer.h"
#include "IEUtility/IETimer.h"
#include "GFG/GFGFileExporter.h"
#include "ASCIIProgressBar.h"
#include <sstream>
#include <iomanip>
#include <locale>

GLFWwindow* OGLVoxelizer::window = nullptr;

OGLVoxelizer::OGLVoxelizer(const VoxelizerOptions& options, 
						   MeshBatch& batch,
						   GL3DTexture& lockTex,
						   StructuredBuffer<IEVector4>& normalArray,
						   StructuredBuffer<IEVector4>& colorArray,
						   StructuredBuffer<VoxelWeightData>& weightArray,
						   Shader& compSplitCount,
						   Shader& compPackVoxels,
						   Shader& compPackVoxelsSkel,
						   Shader& vertVoxelize,
						   Shader& geomVoxelize,
						   Shader& fragVoxelize,
						   Shader& vertVoxelizeSkel,
						   Shader& geomVoxelizeSkel,
						   Shader& fragVoxelizeSkel,
						   Shader& fragVoxelizeCount,
						   bool isSkeletal)
	: options(options)
	, batch(batch)
	, split(batch.DrawCount())
	, objectInfos(batch.DrawCount())
	, totalVoxCount(1)
	, lockTex(lockTex)
	, normalArray(normalArray)
	, colorArray(colorArray)
	, weightArray(weightArray)
	, compSplitCount(compSplitCount)
	, compPackVoxels(compPackVoxels)
	, compPackVoxelsSkel(compPackVoxelsSkel)
	, vertVoxelize(vertVoxelize)
	, geomVoxelize(geomVoxelize)
	, fragVoxelize(fragVoxelize)
	, vertVoxelizeSkel(vertVoxelizeSkel)
	, geomVoxelizeSkel(geomVoxelizeSkel)
	, fragVoxelizeSkel(fragVoxelizeSkel)
	, fragVoxelizeCount(fragVoxelizeCount)
	, isSkeletal(isSkeletal)
	, mipInfo(batch.DrawCount(), MipInfo::EMPTY)
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

	window = glfwCreateWindow(2, 2, "Voxelizer", nullptr, nullptr);
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

double OGLVoxelizer::DetermineSplits(float currentSpan)
{
	auto& aabbBuffer = batch.getDrawBuffer().getAABBBuffer();

	// Timing
	OGLTimer t;
	t.Start();

	// Determine Render Count for Each Object
	compSplitCount.Bind();

	// Uniforms Constants
	glUniform1ui(U_TOTAL_OBJ_COUNT, static_cast<GLuint>(batch.DrawCount()));
	glUniform1f(U_SPAN, currentSpan);
	glUniform1ui(U_GRID_DIM, VOX_3D_TEX_SIZE);
	glUniform1ui(U_VOX_LIMIT, VOX_PACK_LIMITATION);

	// ShaderStorage
	aabbBuffer.BindAsShaderStorageBuffer(LU_AABB);
	split.BindAsShaderStorageBuffer(LU_OBJECT_SPLIT_INFO);
	objectInfos.BindAsShaderStorageBuffer(LU_OBJECT_VOXEL_INFO);

	size_t blockCount = (batch.DrawCount() + BLOCK_SIZE - 1) / BLOCK_SIZE;
	glDispatchCompute(static_cast<GLuint>(blockCount), 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	split.RecieveData(batch.DrawCount());

	for(int i = 0; i < batch.DrawCount(); i++)
	{
		if(mipInfo[i] == MipInfo::EMPTY &&
		   split.CPUData()[i].voxSplit[0] != 0 &&
		   split.CPUData()[i].voxSplit[1] != 0 &&
		   split.CPUData()[i].voxSplit[2] != 0)
		{
			mipInfo[i] = MipInfo::NOT_MIP;
		}
		else if(mipInfo[i] == MipInfo::NOT_MIP &&
				split.CPUData()[i].voxSplit[0] != 0 &&
				split.CPUData()[i].voxSplit[1] != 0 &&
				split.CPUData()[i].voxSplit[2] != 0)
		{
			mipInfo[i] = MipInfo::MIP;
		}
	}

	t.Stop();
	return t.ElapsedMS();
}

double OGLVoxelizer::AllocateVoxelCaches(float currentSpan, uint32_t currentCascade)
{
	GI_LOG("Calculating Allocation Size...");
	OGLTimer timer;
	timer.Start();

	// Segment Size
	float segmentSize = static_cast<float>(VOX_3D_TEX_SIZE) * currentSpan;

	// Shaders Common
	geomVoxelize.Bind();

	// Buffers
	auto& drawBuffer = batch.getDrawBuffer().getDrawParamBuffer();
	auto& aabbBuffer = batch.getDrawBuffer().getAABBBuffer();
	auto& gpuBuffer = batch.getGPUBuffer();

	gpuBuffer.Bind();
	drawBuffer.BindAsDrawIndirectBuffer();
	aabbBuffer.BindAsShaderStorageBuffer(LU_AABB);
	totalVoxCount.BindAsShaderStorageBuffer(LU_TOTAL_VOX_COUNT);
	objectInfos.BindAsShaderStorageBuffer(LU_OBJECT_VOXEL_INFO);
	
	totalVoxCount.Memset(static_cast<uint32_t>(0));

	// Images
	lockTex.BindAsImage(I_LOCK, GL_READ_WRITE);

	// States
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnable(GL_MULTISAMPLE);

	glDepthMask(false);
	glStencilMask(0x0000);
	glColorMask(false, false, false, false);

	GLsizei totalSize = static_cast<GLsizei>(VOX_3D_TEX_SIZE * options.splatRatio);
	glViewport(0, 0, totalSize, totalSize);
	
	// For Each Object
	for(GLuint objIndex = 0; objIndex < batch.DrawCount(); objIndex++)
	{
		// Split Geom to segments
		auto& voxSplit = split.CPUData()[objIndex].voxSplit;
		for(GLuint a = 0; a < voxSplit[0]; a++)
		for(GLuint b = 0; b < voxSplit[1]; b++)
		for(GLuint c = 0; c < voxSplit[2]; c++)
		{
			lockTex.Clear();
			glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

			// Vertex Shader
			vertVoxelize.Bind();
			glUniform1ui(U_OBJ_ID, objIndex);
			glUniform3ui(U_SPLIT_CURRENT, a, b, c);
			glUniform1f(U_SEGMENT_SIZE, segmentSize);
			glUniform1f(U_SPLAT_RATIO, options.splatRatio);
	
			// Fragment Shader
			fragVoxelizeCount.Bind();
			glUniform1ui(U_OBJ_ID, objIndex);
			glUniform3ui(U_SPLIT_CURRENT, a, b, c);
			glUniform1f(U_SEGMENT_SIZE, segmentSize);
			glUniform1f(U_SPAN, currentSpan);
			glUniform1ui(U_TEX_SIZE, VOX_3D_TEX_SIZE);

			glDrawElementsIndirect(GL_TRIANGLES,
								   GL_UNSIGNED_INT,
								   (void*)(objIndex * sizeof(DrawPointIndexed)));
			glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
							GL_SHADER_STORAGE_BARRIER_BIT);
		}
	}
	totalVoxCount.RecieveData(1);
	objectInfos.RecieveData(static_cast<uint32_t>(batch.DrawCount()));

	// Insert to this
	std::memcpy(totalObjInfos.data() + currentCascade * batch.DrawCount() * sizeof(ObjInfo),
				objectInfos.CPUData().data(), batch.DrawCount() * sizeof(ObjInfo));

	uint32_t totalVox = totalVoxCount.CPUData().front();
	voxelNormPos.Resize(totalVox);
	color.Resize(totalVox);
	voxIds.Resize(totalVox);
	if(isSkeletal) weights.Resize(totalVox);

	timer.Stop();

	GI_LOG("Allocation %fms, %dvox", timer.ElapsedMS(), totalVoxCount.CPUData().front());
	GI_LOG("Allocation Size %fmb", (VoxelSize() / 1024.0 / 1024.0));
	GI_LOG("");
	return timer.ElapsedMS();
}

double OGLVoxelizer::GenVoxelWeights()
{
	// TODO: GEOSEDIC VOXEL PAPER
	return 0.0;
}

double OGLVoxelizer::Voxelize(float currentSpan)
{
	GI_LOG("Starting Voxelization...");

	// Timing
	OGLTimer timer; 
	timer.Start();

	// Segment Size
	float segmentSize = static_cast<float>(VOX_3D_TEX_SIZE) * currentSpan;

	// Shaders Common
	if(isSkeletal) geomVoxelizeSkel.Bind();
	else geomVoxelize.Bind();

	// Buffers
	StructuredBuffer<uint32_t> index(1);
	auto& drawBuffer = batch.getDrawBuffer().getDrawParamBuffer();
	auto& aabbBuffer = batch.getDrawBuffer().getAABBBuffer();
	auto& gpuBuffer = batch.getGPUBuffer();

	gpuBuffer.Bind();
	drawBuffer.BindAsDrawIndirectBuffer();
	voxelNormPos.BindAsShaderStorageBuffer(LU_VOXEL_NORM_POS);
	color.BindAsShaderStorageBuffer(LU_VOXEL_COLOR);
	if(isSkeletal) weights.BindAsShaderStorageBuffer(LU_VOXEL_WEIGHT);
	aabbBuffer.BindAsShaderStorageBuffer(LU_AABB);
	voxIds.BindAsShaderStorageBuffer(LU_VOXEL_IDS);
	index.BindAsShaderStorageBuffer(LU_INDEX_CHECK);
	colorArray.BindAsShaderStorageBuffer(LU_COLOR_SPARSE);
	normalArray.BindAsShaderStorageBuffer(LU_NORMAL_SPARSE);
	if(isSkeletal) weightArray.BindAsShaderStorageBuffer(LU_WEIGHT_SPARSE);

	index.Memset(static_cast<uint32_t>(0));

	// Images
	lockTex.BindAsImage(I_LOCK, GL_READ_WRITE);

	// States
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnable(GL_MULTISAMPLE);

	glDepthMask(false);
	glStencilMask(0x0000);
	glColorMask(false, false, false, false);

	GLsizei totalSize = static_cast<GLsizei>(VOX_3D_TEX_SIZE * options.splatRatio);
	glViewport(0, 0, totalSize, totalSize);


	double clearTime = 0.0;
	double voxelizeTime = 0.0;
	double packTime = 0.0;
	double totalClearTime = 0.0;
	double totalVoxelizeTime = 0.0;
	double totalPackTime = 0.0;

	// Buffers
	for(uint32_t objIndex = 0; objIndex < batch.DrawCount(); objIndex++)
	{
		// Split Geom to segments
		auto& voxSplit = split.CPUData()[objIndex].voxSplit;
		for(GLuint a = 0; a < voxSplit[0]; a++)
		for(GLuint b = 0; b < voxSplit[1]; b++)
		for(GLuint c = 0; c < voxSplit[2]; c++)
		{
			//timer.Start();
			lockTex.Clear();
			normalArray.Memset(static_cast<uint32_t>(0));
			colorArray.Memset(static_cast<uint32_t>(0));
			//timer.Stop();
			//clearTime = timer.ElapsedMS();
			//timer.Start();

			glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
		
			const AABBData& objAABB = batch.getDrawBuffer().getAABBBuffer().CPUData()[objIndex];
			GLuint voxDimX, voxDimY, voxDimZ;
			voxDimX = static_cast<GLuint>(std::floor((objAABB.max.getX() - objAABB.min.getX()) / options.span)) + 1;
			voxDimY = static_cast<GLuint>(std::floor((objAABB.max.getY() - objAABB.min.getY()) / options.span)) + 1;
			voxDimZ = static_cast<GLuint>(std::floor((objAABB.max.getZ() - objAABB.min.getZ()) / options.span)) + 1;

			if(voxSplit[0] == 1 &&
			   voxSplit[1] == 1 &&
			   voxSplit[2] == 1)
			{
				voxDimX = std::min<GLuint>(voxDimX, VOX_3D_TEX_SIZE);
				voxDimY = std::min<GLuint>(voxDimY, VOX_3D_TEX_SIZE);
				voxDimZ = std::min<GLuint>(voxDimZ, VOX_3D_TEX_SIZE);
			}
			else
			{
				voxDimX = VOX_3D_TEX_SIZE;
				voxDimY = VOX_3D_TEX_SIZE;
				voxDimZ = VOX_3D_TEX_SIZE;
			}

			// Material Bind
			batch.getDrawBuffer().BindMaterialForDraw(objIndex);

			GLuint isMip = (mipInfo[objIndex] == MipInfo::MIP) ? 1 : 0;
			VoxelizeObject(objIndex, segmentSize, a, b, c, currentSpan);

			//timer.Stop();
			//voxelizeTime = timer.ElapsedMS();
			//timer.Start();

			PackObjectVoxels(objIndex, isMip, voxDimX, voxDimY, voxDimZ,
							 a, b, c);

			//timer.Stop();
			//packTime = timer.ElapsedMS();

			//GI_LOG("Object %d", objIndex);
			//GI_LOG("Texture Clear %f", clearTime);
			//GI_LOG("Voxelize %f", voxelizeTime);
			//GI_LOG("Pack %f", packTime);
			//GI_LOG("------------");
			//totalClearTime += clearTime;
			//totalVoxelizeTime += voxelizeTime;
			//totalPackTime += packTime;
		}
	}
	timer.Stop();
	GI_LOG("Voxelization %fms", timer.ElapsedMS());

	//GI_LOG("Voxelization",);
	//GI_LOG("Total Texture Clear %f", totalClearTime);
	//GI_LOG("Total Voxelize %f", totalVoxelizeTime);
	//GI_LOG("Total Pack %f", totalPackTime);
	//GI_LOG("Grand Total %f", totalPackTime + totalClearTime + totalVoxelizeTime);
	//GI_LOG("------------");

	// Assertion of the Voxel Generation is same as count calculation
	index.RecieveData(1);

	//DEBUG
	totalVoxCount.CPUData().front() = index.CPUData().front();
	assert(index.CPUData().front() == totalVoxCount.CPUData().front());
	return timer.ElapsedMS();
}

void OGLVoxelizer::VoxelizeObject(uint32_t objIndex, float segmentSize,
								  GLuint splitX, GLuint splitY, GLuint splitZ,
								  float currentSpan)
{
	// Vertex Shader
	if(isSkeletal) vertVoxelizeSkel.Bind();
	else vertVoxelize.Bind();
	glUniform1ui(U_OBJ_ID, objIndex);
	glUniform3ui(U_SPLIT_CURRENT, splitX, splitY, splitZ);
	glUniform1f(U_SEGMENT_SIZE, segmentSize);
	glUniform1f(U_SPLAT_RATIO, options.splatRatio);

	// Fragment Shader
	if(isSkeletal) fragVoxelizeSkel.Bind();
	else fragVoxelize.Bind();
	glUniform1ui(U_OBJ_ID, objIndex);
	glUniform3ui(U_SPLIT_CURRENT, splitX, splitY, splitZ);
	glUniform1f(U_SEGMENT_SIZE, segmentSize);
	glUniform1f(U_SPAN, currentSpan);
	glUniform1ui(U_TEX_SIZE, VOX_3D_TEX_SIZE);
	
	glDrawElementsIndirect(GL_TRIANGLES,
						   GL_UNSIGNED_INT,
						   (void*)(objIndex * sizeof(DrawPointIndexed)));

	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
					GL_SHADER_STORAGE_BARRIER_BIT);
}

void OGLVoxelizer::PackObjectVoxels(uint32_t objIndex, GLuint isMip,
									uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ,
									uint32_t splitX, uint32_t splitY, uint32_t splitZ)
{
	if(isSkeletal) compPackVoxelsSkel.Bind();
	else compPackVoxels.Bind();
	glUniform1ui(U_OBJ_TYPE, static_cast<GLuint>(batch.MeshType()));
	glUniform1ui(U_OBJ_ID, objIndex);
	glUniform1ui(U_MAX_CACHE_SIZE, totalVoxCount.CPUData().front());
	glUniform3ui(U_SPLIT_CURRENT, splitX, splitY, splitZ);
	glUniform1ui(U_IS_MIP, isMip);
	glUniform1ui(U_TEX_SIZE, VOX_3D_TEX_SIZE);
	glDispatchCompute(sizeX + 7 / 8,
					  sizeY + 7 / 8,
					  sizeZ + 7 / 8);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | 
					GL_SHADER_STORAGE_BARRIER_BIT);
}

double OGLVoxelizer::FormatToGFG(float currentSpan)
{
	IETimer t;
	t.Start();

	uint32_t totalVox = totalVoxCount.CPUData().front();

	voxelNormPos.RecieveData(totalVox);
	color.RecieveData(totalVox);
	voxIds.RecieveData(totalVox);
	if(isSkeletal) weights.RecieveData(totalVox);

	std::vector<uint8_t> data(totalVox * (sizeof(VoxelNormPos) +
							  sizeof(VoxelColorData) +
							  sizeof(VoxelIds) +
							  (isSkeletal ? sizeof(VoxelWeightData) : 0)));

	std::memcpy(data.data(),
				reinterpret_cast<uint8_t*>(voxelNormPos.CPUData().data()),
				totalVox * sizeof(VoxelNormPos));
	std::memcpy(data.data() + totalVox * sizeof(VoxelNormPos),
				reinterpret_cast<uint8_t*>(voxIds.CPUData().data()),
				totalVox * sizeof(VoxelIds));
	std::memcpy(data.data() + totalVox * (sizeof(VoxelNormPos) + sizeof(VoxelIds)),
				reinterpret_cast<uint8_t*>(color.CPUData().data()),
				totalVox * sizeof(VoxelColorData));
	if(isSkeletal)
	{
		std::memcpy(data.data() + totalVox * (sizeof(VoxelNormPos) +
					sizeof(VoxelColorData) + sizeof(VoxelIds)),
					reinterpret_cast<uint8_t*>(weights.CPUData().data()),
					totalVox * sizeof(VoxelWeightData));
	}

	assert(GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)] == sizeof(VoxelNormPos));
	assert(GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)] == sizeof(VoxelIds));
	assert(GFGDataTypeByteSize[static_cast<int>(GFGDataType::UNORM8_4)] == sizeof(VoxelColorData));
	assert(GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)] == sizeof(VoxelWeightData));

	std::vector<GFGVertexComponent> components =
	{
		GFGVertexComponent
		{	// NormPos
			GFGDataType::UINT32_2,
			GFGVertexComponentLogic::POSITION,
			0,
			0,
			GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)]
		},
		GFGVertexComponent
		{	// Ids
			GFGDataType::UINT32_2,
			GFGVertexComponentLogic::NORMAL,
			totalVox * sizeof(VoxelNormPos),
			0,
			GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)]
		},
		GFGVertexComponent
		{	// Color
			GFGDataType::UNORM8_4,
			GFGVertexComponentLogic::COLOR,
			totalVox * (sizeof(VoxelNormPos) + sizeof(VoxelIds)),
			0,
			GFGDataTypeByteSize[static_cast<int>(GFGDataType::UNORM8_4)]

		},
		GFGVertexComponent
		{	// Weights
			GFGDataType::UINT32_2,
			GFGVertexComponentLogic::WEIGHT,
			totalVox * (sizeof(VoxelNormPos) + sizeof(VoxelIds) + sizeof(VoxelColorData)),
			0,
			GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)]
		}
	};

	if(!isSkeletal) components.pop_back();

	GFGMeshHeaderCore meshHeader = {0};
	meshHeader.indexCount = 0;
	meshHeader.indexSize = 0;
	meshHeader.topology = GFGTopology::POINT;
	meshHeader.vertexCount = totalVoxCount.CPUData().front();
	meshHeader.aabb.min[0] = currentSpan;

	fileOut.AddMesh(0, components, meshHeader, data);

	t.Stop();
	return t.ElapsedMilliS();
}

double OGLVoxelizer::Write(const std::string& fileName)
{
	IETimer t;
	t.Start();

	// Send Last one as Component
	GFGMeshHeader meshObj;
	meshObj.headerCore = {0};
	meshObj.headerCore.vertexCount = batch.DrawCount();
	meshObj.headerCore.componentCount = options.cascadeCount;
	meshObj.headerCore.topology = GFGTopology::POINT;
	for(unsigned int i = 0; i < options.cascadeCount; i++)
	{
		meshObj.components.emplace_back();
		meshObj.components.back().dataType = GFGDataType::UINT32_2;
		meshObj.components.back().internalOffset = 0;
		meshObj.components.back().stride = GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)];
		meshObj.components.back().logic = GFGVertexComponentLogic::POSITION;
		meshObj.components.back().startOffset = i * GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)] * batch.DrawCount();
	}

	// Component Mesh
	fileOut.AddMesh(0,
					meshObj.components,
					meshObj.headerCore,
					totalObjInfos);
	
	// GFG File Writing
	std::ofstream fileStream;
	fileStream.open(fileName, std::ofstream::binary);
	GFGFileWriterSTL fw(fileStream);

	fileOut.Write(fw);
	fileStream.close();

	t.Stop();
	return t.ElapsedMilliS();
}

uint64_t OGLVoxelizer::VoxelSize()
{
	uint64_t result = totalVoxCount.CPUData().front() * 
					  (sizeof(VoxelNormPos) + sizeof(VoxelColorData) +
					  sizeof(VoxelIds) + ((isSkeletal) ? sizeof(VoxelWeightData) : 0));
	return result;
}

void OGLVoxelizer::Start()
{
	totalObjInfos.resize(batch.DrawCount() * options.cascadeCount * sizeof(ObjInfo));

	for(uint32_t i = 0; i < options.cascadeCount; i++)
	{
		// Timing Voxelization Process
		double totalTime = 0.0;
		float currentSpan = options.span * (1 << i);

		// Split Determination
		totalTime += DetermineSplits(currentSpan);

		// Voxel Count Determination
		totalTime += AllocateVoxelCaches(currentSpan, i);

		// Actual Voxelization
		totalTime += Voxelize(currentSpan);

		// Weight Generation
		if(isSkeletal) GenVoxelWeights();

		// Sending to GFG File
		totalTime += FormatToGFG(currentSpan);

		GI_LOG("Cascade#%d %fms, Span %f", i, totalTime, currentSpan);
		GI_LOG("------------------------------------");
	}
}