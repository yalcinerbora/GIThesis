#include <fstream>
#include "GFG/GFGFileLoader.h"

#include "DeferredRenderer.h"
#include "IEUtility/IEMath.h"
#include "Scene.h"
#include "Globals.h"
#include "Camera.h"
#include "RectPrism.h"
#include "DrawBuffer.h"
#include "MeshBatchSkeletal.h"
#include "BindPoints.h"

LightDrawBuffer::LightDrawBuffer()
{
	std::ifstream stream(LightAOIFileName, std::ios_base::in | std::ios_base::binary);
	GFGFileReaderSTL stlFileReader(stream);
	GFGFileLoader gfgFile(&stlFileReader);
	gfgFile.ValidateAndOpen();

	assert(gfgFile.Header().meshes.size() == LightTypeCount);
	std::vector<uint8_t> vData(gfgFile.AllMeshVertexDataSize());
	std::vector<uint8_t> viData(gfgFile.AllMeshIndexDataSize());
	gfgFile.AllMeshVertexData(vData.data());
	gfgFile.AllMeshIndexData(viData.data());

	// Generate DrawPoint Indexed
	uint32_t vOffset = 0, viOffset = 0;
	for(int i = 0; i < LightTypeCount; i++)
	{
		const GFGMeshHeader& mesh = gfgFile.Header().meshes[i];
		assert(mesh.headerCore.indexSize == sizeof(uint32_t));

		lightDrawParams[i].baseInstance = 0;
		lightDrawParams[i].baseVertex = vOffset;
		lightDrawParams[i].firstIndex = viOffset;
		lightDrawParams[i].count = static_cast<uint32_t>(mesh.headerCore.indexCount);
		lightDrawParams[i].instanceCount = 0;

		vOffset += static_cast<uint32_t>(mesh.headerCore.vertexCount);
		viOffset += static_cast<uint32_t>(mesh.headerCore.indexCount);
	}
	
	// Offset and Total Buffer Size Calculation
	size_t totalSize = 0;
	// DP
	totalSize = DeviceOGLParameters::AlignOffset(totalSize, 4);
	drawOffset = totalSize;
	totalSize += lightDrawParams.size() * sizeof(DrawPointIndexed);
	// Vertex
	vertexOffset = totalSize;
	totalSize += vOffset * sizeof(float) * 3;
	// Index
	totalSize = DeviceOGLParameters::AlignOffset(totalSize, sizeof(uint32_t));
	indexOffset = totalSize;
	totalSize += viOffset * sizeof(uint32_t);

	// Incorporate Offset Into First Index
	for(int i = 0; i < LightTypeCount; i++)
	{
		lightDrawParams[i].firstIndex += static_cast<uint32_t>(indexOffset / sizeof(uint32_t));
	}

	// Copy To Buffer
	gpuData.Resize(totalSize);
	auto& cpuImage = gpuData.CPUData();
	std::copy(reinterpret_cast<uint8_t*>(lightDrawParams.data()),
			  reinterpret_cast<uint8_t*>(lightDrawParams.data() + lightDrawParams.size()),
			  cpuImage.data() + drawOffset);
	std::copy(vData.data(), 
			  vData.data() + vOffset * sizeof(float) * 3,
			  cpuImage.data() + vertexOffset);
	std::copy(viData.data(), viData.data() + viOffset * sizeof(uint32_t),			  
			  cpuImage.data() + indexOffset);
	gpuData.SendData();

	// Create VAO
	// PostProcess VAO
	glGenVertexArrays(1, &lightVAO);
	glBindVertexArray(lightVAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuData.getGLBuffer());

	// Pos
	glBindVertexBuffer(0, gpuData.getGLBuffer(),
					   static_cast<GLintptr>(vertexOffset), sizeof(float) * 3);
	glEnableVertexAttribArray(IN_POS);
	glVertexAttribFormat(IN_POS, 3, GL_FLOAT, false, 0);
	glVertexAttribBinding(IN_POS, 0);
}

LightDrawBuffer::~LightDrawBuffer()
{
	glDeleteVertexArrays(1, &lightVAO);
}

void LightDrawBuffer::AttachSceneLights(SceneLights& sceneLights)
{
	uint32_t instanceOffset = 0;
	for(int i = 0; i < LightTypeCount; i++)
	{
		LightType currentType = static_cast<LightType>(i);
		lightDrawParams[i].baseInstance = instanceOffset;
		lightDrawParams[i].instanceCount = sceneLights.getLightCount(currentType);
		instanceOffset += lightDrawParams[i].instanceCount;
	}
	assert(instanceOffset == sceneLights.getLightCount());
	std::copy(reinterpret_cast<uint8_t*>(lightDrawParams.data()),
			  reinterpret_cast<uint8_t*>(lightDrawParams.data() + lightDrawParams.size()),
			  gpuData.CPUData().data() + drawOffset);
	gpuData.SendSubData(static_cast<uint32_t>(drawOffset),
						static_cast<uint32_t>(lightDrawParams.size() * sizeof(DrawPointIndexed)));

	// VAO
	GLintptr bufferOffset = sceneLights.getLightIndexOffset();
	glBindVertexArray(lightVAO);
	glBindVertexBuffer(1, sceneLights.getGLBuffer(), bufferOffset, sizeof(uint32_t));
	glVertexBindingDivisor(1, 1);
	glEnableVertexAttribArray(IN_LIGHT_INDEX);
	glVertexAttribIFormat(IN_LIGHT_INDEX, 1, GL_UNSIGNED_INT, 0);
	glVertexAttribBinding(IN_LIGHT_INDEX, 1);
}

void LightDrawBuffer::BindVAO()
{
	glBindVertexArray(lightVAO);
}

void LightDrawBuffer::BindDrawIndirectBuffer()
{
	gpuData.BindAsDrawIndirectBuffer();
}

void LightDrawBuffer::DrawCall()
{
	static_assert(sizeof(GLintptr) == sizeof(void*), "Unappropirate GL Offset Parameter");
	GLintptr offset = static_cast<GLintptr>(drawOffset);
	glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT,
								(void*)(offset), LightTypeCount, sizeof(DrawPointIndexed));

}

void DeferredRenderer::BindInvFrameTransform(GLuint bindingPoint)
{
	gpuData.BindAsUniformBuffer(bindingPoint, 
								static_cast<GLuint>(iOffset), 
								sizeof(InvFrameTransform));
}

void DeferredRenderer::BindFrameTransform(GLuint bindingPoint)
{
	gpuData.BindAsUniformBuffer(bindingPoint,
								static_cast<GLuint>(fOffset), 
								sizeof(FrameTransformData));
}

void DeferredRenderer::UpdateFTransformBuffer()
{
	std::copy(reinterpret_cast<uint8_t*>(&fTransform),
			  reinterpret_cast<uint8_t*>(&fTransform) + sizeof(FrameTransformData),
			  gpuData.CPUData().data() + fOffset);
	gpuData.SendSubData(static_cast<uint32_t>(fOffset),
						sizeof(FrameTransformData));
}

void DeferredRenderer::UpdateInvFTransformBuffer()
{
	std::copy(reinterpret_cast<uint8_t*>(&ifTransform),
			  reinterpret_cast<uint8_t*>(&ifTransform) + sizeof(InvFrameTransform),
			  gpuData.CPUData().data() + iOffset);
	gpuData.SendSubData(static_cast<uint32_t>(iOffset), 
						sizeof(InvFrameTransform));
}

DeferredRenderer::DeferredRenderer()
	// Geom Write
	: fragGBufferWrite(ShaderType::FRAGMENT, "Shaders/GWriteGeneric.frag")
	// Depth Prepass Shaders
	, fragDPass(ShaderType::FRAGMENT, "Shaders/DPass.frag")
	// Light Pass
	, vertLightPass(ShaderType::VERTEX, "Shaders/LightPass.vert")
	, fragLightPass(ShaderType::FRAGMENT, "Shaders/LightPass.frag")
	// Post process
	, vertPPGeneric(ShaderType::VERTEX, "Shaders/PProcessGeneric.vert")
	, fragLightApply(ShaderType::FRAGMENT, "Shaders/PPLightPresent.frag")
	, fragPPGeneric(ShaderType::FRAGMENT, "Shaders/PProcessGeneric.frag")
	, fragPPGBuffer(ShaderType::FRAGMENT, "Shaders/PProcessGBuff.frag")
	, fragPPShadowMap(ShaderType::FRAGMENT, "Shaders/PProcessShadowMap.frag")
	// Shadow Map
	, fragShadowMap(ShaderType::FRAGMENT, "Shaders/ShadowMap.frag")
	, computeHierZ(ShaderType::COMPUTE, "Shaders/HierZ.glsl")
	// GBuffer
	, gBuffer(GBuffWidth, GBuffHeight)
	, fTransform{IEMatrix4x4::IdentityMatrix, IEMatrix4x4::IdentityMatrix}
	, ifTransform{IEMatrix4x4::IdentityMatrix,
				  IEVector4::ZeroVector,
				  IEVector4::ZeroVector,
				  {0, 0, 0, 0},
				  IEVector4::ZeroVector}
	, fOffset(0)
	, iOffset(0)
	, lightIntensityTex(0)
	, lightIntensityFBO(0)
	, sRGBEndTex(0)
	, sRGBEndFBO(0)
	, postProcessTriVao(0)
	, flatSampler(0)
	, linearSampler(0)
	, shadowMapSampler(0)
{
	// GBuffer Write
	vertGBufferWrite[static_cast<int>(MeshBatchType::RIGID)] = Shader(ShaderType::VERTEX, "Shaders/GWriteGeneric.vert");
	vertGBufferWrite[static_cast<int>(MeshBatchType::SKELETAL)] = Shader(ShaderType::VERTEX, "Shaders/GWriteSkeletal.vert");
	// Depth Prepass
	vertDPass[static_cast<int>(MeshBatchType::RIGID)] = Shader(ShaderType::VERTEX, "Shaders/DPass.vert");
	vertDPass[static_cast<int>(MeshBatchType::SKELETAL)] = Shader(ShaderType::VERTEX, "Shaders/DPassSkeletal.vert");
	// Shadow Map
	vertShadowMap[static_cast<int>(MeshBatchType::RIGID)] = Shader(ShaderType::VERTEX, "Shaders/ShadowMap.vert");
	vertShadowMap[static_cast<int>(MeshBatchType::SKELETAL)] = Shader(ShaderType::VERTEX, "Shaders/ShadowMapSkeletal.vert");
	// Light Geometry
	geomShadowMap[static_cast<int>(LightType::POINT)] = Shader(ShaderType::GEOMETRY, "Shaders/ShadowMapP.geom");
	geomShadowMap[static_cast<int>(LightType::DIRECTIONAL)] = Shader(ShaderType::GEOMETRY, "Shaders/ShadowMapD.geom");

	// Light Intensity Tex
	glGenTextures(1, &lightIntensityTex);
	glGenFramebuffers(1, &lightIntensityFBO);

	glBindTexture(GL_TEXTURE_2D, lightIntensityTex);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, GBuffWidth, GBuffHeight);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, lightIntensityFBO);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, lightIntensityTex, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, gBuffer.getDepthGL(), 0);
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

	// SRGB Tex
	glGenTextures(1, &sRGBEndTex);
	glGenFramebuffers(1, &sRGBEndFBO);
	glBindTexture(GL_TEXTURE_2D, sRGBEndTex);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_SRGB8_ALPHA8, GBuffWidth, GBuffHeight);

	glBindFramebuffer(GL_FRAMEBUFFER, sRGBEndFBO);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, sRGBEndTex, 0);
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

	// Validate sRGB Encoding
	GLint encoding;
	glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, 
										  GL_COLOR_ATTACHMENT0,
										  GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING, 
										  &encoding);
	assert(encoding == GL_SRGB);
	
	// Pack to GPU Buffer
	size_t totalSize = 0;
	// Light AOI Vertices
	postTriOffset = totalSize;
	totalSize += 6 * sizeof(float);
	// Frame Transform 
	totalSize = DeviceOGLParameters::UBOAlignOffset(totalSize);
	fOffset = totalSize;
	totalSize += sizeof(FrameTransformData);
	// Inv Frame Transform
	totalSize = DeviceOGLParameters::UBOAlignOffset(totalSize);
	iOffset = totalSize;
	totalSize += sizeof(InvFrameTransform);
	// Copy Data
	gpuData.Resize(totalSize);
	auto& cpuImage = gpuData.CPUData();
	std::copy(reinterpret_cast<const uint8_t*>(postProcessTriData),
			  reinterpret_cast<const uint8_t*>(postProcessTriData) + 6 * sizeof(float),
			  cpuImage.data() + postTriOffset);
	gpuData.SendSubData(static_cast<uint32_t>(postTriOffset), 6 * sizeof(float));
	
	// PostProcess VAO
	glGenVertexArrays(1, &postProcessTriVao);
	glBindVertexArray(postProcessTriVao);
	glBindVertexBuffer(0, gpuData.getGLBuffer(), postTriOffset, sizeof(float) * 2);
	glEnableVertexAttribArray(IN_POS);
	glVertexAttribFormat(IN_POS, 2, GL_FLOAT, false, 0);
	glVertexAttribBinding(IN_POS, 0);

	// Samplers
	glGenSamplers(1, &flatSampler);
	glGenSamplers(1, &linearSampler);
	glGenSamplers(1, &shadowMapSampler);

	glSamplerParameteri(flatSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glSamplerParameteri(flatSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glSamplerParameteri(linearSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glSamplerParameteri(linearSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
	//glSamplerParameteri(shadowMapSampler, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	//glSamplerParameteri(shadowMapSampler, GL_TEXTURE_COMPARE_FUNC, GL_LESS);
	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

	GLfloat col[] = { 1.0f, 0.0f, 0.0f, 0.0f };
	glSamplerParameterfv(shadowMapSampler, GL_TEXTURE_BORDER_COLOR, col);
}

DeferredRenderer::~DeferredRenderer()
{
	glDeleteTextures(1, &lightIntensityTex);
	glDeleteFramebuffers(1, &lightIntensityFBO);
	glDeleteTextures(1, &sRGBEndTex);	
	glDeleteFramebuffers(1, &sRGBEndFBO);
	glDeleteVertexArrays(1, &postProcessTriVao);
	glDeleteSamplers(1, &flatSampler);
	glDeleteSamplers(1, &linearSampler);
	glDeleteSamplers(1, &shadowMapSampler);
}

GBuffer& DeferredRenderer::getGBuffer()
{
	return gBuffer;
}

GLuint DeferredRenderer::getLightIntensityBufferGL()
{
	return lightIntensityTex;
}

void DeferredRenderer::GenerateShadowMaps(SceneI& scene, const Camera& camera, bool doTiming)
{
	if(doTiming) oglTimer.Start();

	SceneLights& sLights = scene.getSceneLights();
	uint32_t lightCount = static_cast<uint32_t>(sLights.getLightCount());
	
	// State
	// Rendering with polygon offset to eliminate shadow acne
	glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
	glColorMask(true, false, false, false);
	glDepthMask(true);
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glViewport(0, 0, LightDrawBuffer::ShadowMapWH, LightDrawBuffer::ShadowMapWH);

	// Buffers
	sLights.GenerateMatrices(camera);
	sLights.SendVPMatricesToGPU();
	sLights.BindViewProjectionMatrices(LU_LIGHT_MATRIX);
	
	// Shaders
	fragShadowMap.Bind();

	// Render Loop
	for(uint32_t i = 0; i < lightCount; i++)
	{
		// FBO Bind and render calls
		sLights.BindLightFramebuffer(i);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Skip if this light do not cast shadows
		if(!sLights.getLightCastShadow(i)) continue;

		// Draw Batches
		const std::vector<MeshBatchI*>& batches = scene.getBatches();
		for(size_t j = 0; j < batches.size(); j++)
		{
			if(batches[j]->DrawCount() == 0) continue;

			// Batch Related Shaders
			vertShadowMap[static_cast<int>(batches[j]->MeshType())].Bind();

			// Skeletal Batch Joint Transform Bind
			if(batches[j]->MeshType() == MeshBatchType::SKELETAL)
			{
				auto batchPtr = static_cast<MeshBatchSkeletal*>(batches[j]);
				batchPtr->getJointTransforms().BindAsShaderStorageBuffer(LU_JOINT_TRANS);
			}

			VertexBuffer& currentVertexBuffer = batches[j]->getVertexBuffer();
			DrawBuffer& currentDrawBuffer = batches[j]->getDrawBuffer();

			// VAO Binds
			currentVertexBuffer.Bind();
			currentDrawBuffer.BindAsDrawIndirectBuffer();

			// Type Related Binds
			LightType t = sLights.getLightType(i);
			geomShadowMap[static_cast<int>(t)].Bind();
			switch(t)
			{				
				case LightType::POINT:
					// Base poly offset gives ok results on point lights
					glPolygonOffset(2.64f, 512.0f);
					break;
				case LightType::DIRECTIONAL:
					// Higher offset req since camera span is large
					glPolygonOffset(6.12f, 2048.0f);
					//glPolygonOffset(6.12f, static_cast<float>(LightDrawBuffer::ShadowMapWH)); 
					break;
			}
			glUniform1ui(U_LIGHT_ID, static_cast<GLuint>(i));

			currentDrawBuffer.BindModelTransform(LU_MTRANSFORM);
			
			// Draw Call
			currentDrawBuffer.DrawCallMulti();
		}
	}

	glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
	glFlush();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Hierarchical z-buffers
	GLuint lightTex = scene.getSceneLights().getShadowTextureArrayView();
	computeHierZ.Bind();
	for(unsigned int i = 0; i < LightDrawBuffer::ShadowMapMipCount - 1; i++)
	{
		// Reduce Entire Level at once
		// TODO:
		GLuint depthSize = LightDrawBuffer::ShadowMapWH >> (i + 1);
		GLuint totalPixelCount = (depthSize * depthSize) * lightCount * 6;
		
		glUniform1ui(U_DEPTH_SIZE, depthSize);
		glUniform1ui(U_PIX_COUNT, totalPixelCount);
		
		glBindImageTexture(I_DEPTH_READ, lightTex, i, true, 0, GL_READ_ONLY, GL_R32F);
		glBindImageTexture(I_DEPTH_WRITE, lightTex, i + 1, true, 0, GL_WRITE_ONLY, GL_R32F);
				
		// Dispatch
		unsigned int gridSize = (totalPixelCount + 256 - 1) / 256;
		glDispatchCompute(gridSize, 1, 1);
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	}

	glDisable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(0.0f, 0.0f);

	if(doTiming)
	{
		oglTimer.Stop();
		shadowMapTime = oglTimer.ElapsedMS();
	}
}

void DeferredRenderer::GPass(SceneI& scene, const Camera& camera, bool doTiming)
{
	if(doTiming) oglTimer.Start();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);
	
	// With Depth Prepass
	glDepthMask(false);
	glDepthFunc(GL_EQUAL);
	glColorMask(true, true, true, true);
	
	// Without Depth Prepass
	glDepthMask(true);
	glDepthFunc(GL_LEQUAL);
	
	// Camera Transform
	BindFrameTransform(U_FTRANSFORM);

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	fragGBufferWrite.Bind();

	// DrawCall
	// Draw Batches
	const std::vector<MeshBatchI*>& batches = scene.getBatches();
	for(unsigned int i = 0; i < batches.size(); i++)
	{
		if(batches[i]->DrawCount() == 0) continue;

		vertGBufferWrite[static_cast<int>(batches[i]->MeshType())].Bind();
		if(batches[i]->MeshType() == MeshBatchType::SKELETAL)
		{
			MeshBatchSkeletal* batchPtr = static_cast<MeshBatchSkeletal*>(batches[i]);
			batchPtr->getJointTransforms().BindAsShaderStorageBuffer(LU_JOINT_TRANS);
		}
		
		VertexBuffer& currentVertexBuffer = batches[i]->getVertexBuffer();
		DrawBuffer& currentDrawBuffer = batches[i]->getDrawBuffer();

		currentVertexBuffer.Bind();
		currentDrawBuffer.BindAsDrawIndirectBuffer();
		currentDrawBuffer.BindModelTransform(LU_MTRANSFORM);

		for(unsigned int j = 0; j < batches[i]->DrawCount(); j++)
		{
			currentDrawBuffer.BindMaterialForDraw(j);
			currentDrawBuffer.DrawCallSingle(j);
		}
	}

	if(doTiming)
	{
		oglTimer.Stop();
		gPassTime = oglTimer.ElapsedMS();
	}
}

void DeferredRenderer::ClearLI(const IEVector3& ambientColor)
{
	glBindFramebuffer(GL_FRAMEBUFFER, lightIntensityFBO);
	glClearColor(ambientColor.getX(), ambientColor.getY(), ambientColor.getZ(), 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void DeferredRenderer::LightPass(SceneI& scene, const Camera& camera, bool doTiming)
{
	if(doTiming) oglTimer.Start();

	// Light pass
	// Texture Binds
	gBuffer.BindAsTexture(T_COLOR, RenderTargetLocation::COLOR);
	gBuffer.BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);
	gBuffer.BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
	glBindSampler(T_COLOR, flatSampler);
	glBindSampler(T_NORMAL, flatSampler);
	glBindSampler(T_DEPTH, flatSampler);

	// ShadowMap Binds
	BindShadowMaps(scene);
	BindLightBuffers(scene);

	// Frame Transforms
	RefreshInvFTransform(scene, camera, GBuffWidth, GBuffHeight);
	BindInvFrameTransform(U_INVFTRANSFORM);
	BindFrameTransform(U_FTRANSFORM);
	
	// Bind LightIntensity Buffer as framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, lightIntensityFBO);
	glViewport(0, 0, GBuffWidth, GBuffHeight);

	// Shader and Shader Uniform Binds
	Shader::Unbind(ShaderType::GEOMETRY);
	vertLightPass.Bind();
	fragLightPass.Bind();
	glUniform1ui(U_SHADOW_MIP_COUNT, static_cast<GLuint>(LightDrawBuffer::ShadowMapMipCount));
	glUniform1ui(U_SHADOW_MAP_WH, static_cast<GLuint>(LightDrawBuffer::ShadowMapWH));

	// Open Additive Blending
	// Intensity of different lights will be added
	// Only render backfaces to eliminate multi fragment from light enclosure objects
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_CLAMP);
	glFrontFace(GL_CW);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glDepthFunc(GL_GREATER);
	glDepthMask(false);

	// Bind And Draw
	lightAOI.BindDrawIndirectBuffer();
	lightAOI.BindVAO();
	lightAOI.DrawCall();

	// Restore States
	glFrontFace(GL_CCW);
	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glDisable(GL_DEPTH_CLAMP);

	if(doTiming)
	{
		oglTimer.Stop();
		lPassTime = oglTimer.ElapsedMS();
	}
}

void DeferredRenderer::DPass(SceneI& scene, const Camera& camera, bool doTiming)
{
	if(doTiming) oglTimer.Start();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);
	glDepthMask(true);
	glColorMask(false, false, false, false);
	glDepthFunc(GL_LEQUAL);	
	
	// Camera Transform
	BindFrameTransform(U_FTRANSFORM);

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	fragDPass.Bind();

	// DrawCall
	// Draw Batches
	const std::vector<MeshBatchI*>& batches = scene.getBatches();
	for(unsigned int i = 0; i < batches.size(); i++)
	{
		if(batches[i]->DrawCount() == 0) continue;

		vertDPass[static_cast<int>(batches[i]->MeshType())].Bind();
		if(batches[i]->MeshType() == MeshBatchType::SKELETAL)
		{
			MeshBatchSkeletal* batchPtr = static_cast<MeshBatchSkeletal*>(batches[i]);
			batchPtr->getJointTransforms().BindAsShaderStorageBuffer(LU_JOINT_TRANS);
		}

		VertexBuffer& currentVertexBuffer = batches[i]->getVertexBuffer();
		DrawBuffer& currentDrawBuffer = batches[i]->getDrawBuffer();

		currentVertexBuffer.Bind();
		currentDrawBuffer.BindAsDrawIndirectBuffer();
		currentDrawBuffer.BindModelTransform(LU_MTRANSFORM);

		currentDrawBuffer.DrawCallMulti();
	}

	if(doTiming)
	{
		oglTimer.Stop();
		dPassTime = oglTimer.ElapsedMS();
	}
}

void DeferredRenderer::Present(const Camera& camera, bool doTiming)
{	
	if(doTiming) oglTimer.Start();

	// Render to main framebuffer as post process
	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	vertPPGeneric.Bind();
	fragLightApply.Bind();

	// Texture Binds
	gBuffer.BindAsTexture(T_COLOR, RenderTargetLocation::COLOR);
	glActiveTexture(GL_TEXTURE0 + T_INTENSITY);
	glBindTexture(GL_TEXTURE_2D, lightIntensityTex);
	glBindSampler(T_COLOR, flatSampler);
	glBindSampler(T_INTENSITY, flatSampler);

	// FBO
	glBindFramebuffer(GL_FRAMEBUFFER, sRGBEndFBO);
	glViewport(0, 0, 
			   static_cast<GLsizei>(GBuffWidth),
			   static_cast<GLsizei>(GBuffHeight));

	// States
	glEnable(GL_FRAMEBUFFER_SRGB);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glDepthMask(false);
	glColorMask(true, true, true, true);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_MULTISAMPLE);

	// Bind Post Process VAO
	glBindVertexArray(postProcessTriVao);

	// DrawCall
	glDrawArrays(GL_TRIANGLES, 0, 3);
	
	// Passthrough to Default FBO
	//
	// SRGB Texture
	glActiveTexture(GL_TEXTURE0 + T_COLOR);
	glBindTexture(GL_TEXTURE_2D, sRGBEndTex);
	glBindSampler(T_COLOR, linearSampler);

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	vertPPGeneric.Bind();
	fragPPGeneric.Bind();

	// Default FBO
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(camera.width),
			   static_cast<GLsizei>(camera.height));
	
	// States
	glDisable(GL_FRAMEBUFFER_SRGB);
	glClear(GL_COLOR_BUFFER_BIT);

	// VAO
	glBindVertexArray(postProcessTriVao);

	// Draw
	glDrawArrays(GL_TRIANGLES, 0, 3);

	if(doTiming)
	{
		oglTimer.Stop();
		mergeTime = oglTimer.ElapsedMS();
	}
}

void DeferredRenderer::RefreshInvFTransform(SceneI& scene,
											const Camera& camera,
											GLsizei width,
											GLsizei height)
{
	FrameTransformData ft = camera.GenerateTransform();

	float depthRange[2];
	glGetFloatv(GL_DEPTH_RANGE, depthRange);
	ifTransform = InvFrameTransform
	{
		ft.view.Inverse() * ft.projection.Inverse(),
		IEVector4(camera.pos, scene.getSceneLights().getCascadeLength(camera.far)),
		IEVector4((camera.centerOfInterest - camera.pos).NormalizeSelf()),
		{0, 0, static_cast<unsigned int>(width), static_cast<unsigned int>(height)},
		{depthRange[0], depthRange[1], 0.0f, 0.0f}
	};
	UpdateInvFTransformBuffer();
}

void DeferredRenderer::PopulateGBuffer(SceneI& scene, const Camera& camera, bool doTiming)
{
	// Bind FBO
	gBuffer.BindAsFBO();
	gBuffer.AlignViewport();

	// Clear
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glDepthMask(true);
	glColorMask(true, true, true, true);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);


	// Depth Pre-Pass
	//DPass(scene, camera, doTiming);

	// Actual Render
	// G Pass
	GPass(scene, camera, doTiming);
}

void DeferredRenderer::Render(SceneI& scene, const Camera& camera, bool directLight,
							  const IEVector3& ambientColor, bool doTiming)
{
	// Camera Transform Update
	fTransform = camera.GenerateTransform();
	UpdateFTransformBuffer();

	// Send new Light Params
	scene.getSceneLights().SendLightDataToGPU();

	// Shadow Map Generation
	GenerateShadowMaps(scene, camera, doTiming);

	// GPass
	PopulateGBuffer(scene, camera, doTiming);
	
	// Clear LI with ambient color
	ClearLI(ambientColor);

	// Light Pass
	if(directLight)
	{
		LightPass(scene, camera, doTiming);
	}

	// Light Intensity Merge
	Present(camera, doTiming);

	// All Done!
}

void DeferredRenderer::ShowTexture(const Camera& camera, GLuint tex)//, RenderTargetLocation location)
{
	// Only Draw Color Buffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(camera.width),
			   static_cast<GLsizei>(camera.height));
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	glDisable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LESS);
	glDepthMask(true);
	glColorMask(true, true, true, true);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	vertPPGeneric.Bind();
	fragPPGeneric.Bind();

	// Texture
	glActiveTexture(GL_TEXTURE0 + T_COLOR);
	glBindTexture(GL_TEXTURE_2D, tex);
	glBindSampler(T_COLOR, linearSampler);

	// VAO
	glBindVertexArray(postProcessTriVao);

	// Draw
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void DeferredRenderer::ShowGBufferTexture(const Camera& camera, RenderScheme scheme)
{
	// Only Draw Color Buffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(camera.width),
			   static_cast<GLsizei>(camera.height));
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	glDisable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LESS);
	glDepthMask(true);
	glColorMask(true, true, true, true);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	vertPPGeneric.Bind();
	fragPPGBuffer.Bind();

	// Uniforms (frag shader)
	glUniform2f(U_NEAR_FAR, camera.near, camera.far);
	glUniform1ui(U_RENDER_MODE, static_cast<unsigned int>(scheme));

	// Texture
	glActiveTexture(GL_TEXTURE0 + T_COLOR);
	if(scheme == RenderScheme::G_NORMAL)
		glBindTexture(GL_TEXTURE_2D, gBuffer.getColorGL());
	else if(scheme == RenderScheme::G_DIFF_ALBEDO ||
			scheme == RenderScheme::G_SPEC_ALBEDO)
		glBindTexture(GL_TEXTURE_2D, gBuffer.getColorGL());
	glBindSampler(T_COLOR, linearSampler);
	if(scheme == RenderScheme::G_DEPTH)
	{
		glBindTexture(GL_TEXTURE_2D, gBuffer.getDepthGL());
		glActiveTexture(GL_TEXTURE0 + T_NORMAL);
		glBindSampler(T_NORMAL, linearSampler);
	}

	// VAO
	glBindVertexArray(postProcessTriVao);

	// Draw
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void DeferredRenderer::ShowShadowMap(const Camera& camera,
									 SceneI& s,
									 int lightId, int layer)
{
	// Only Draw Color Buffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(camera.width),
			   static_cast<GLsizei>(camera.height));
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	glDisable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LESS);
	glDepthMask(true);
	glColorMask(true, true, true, true);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	vertPPGeneric.Bind();
	fragPPShadowMap.Bind();

	// Uniforms (frag shader)
	if(s.getSceneLights().getLightType(lightId) == LightType::DIRECTIONAL)
		glUniform2f(U_NEAR_FAR, 0.0f, 0.0f);
	else if(s.getSceneLights().getLightType(lightId) == LightType::POINT)
		glUniform2f(U_NEAR_FAR, SceneLights::PointLightNear, s.getSceneLights().getLightRadius(lightId));
	glUniform1ui(U_LIGHT_ID, static_cast<unsigned int>(SceneLights::CubeSide * lightId + layer));

	// Texture
	glActiveTexture(GL_TEXTURE0 + T_COLOR);
	glBindTexture(GL_TEXTURE_2D_ARRAY, s.getSceneLights().getShadowTextureArrayView());
	glBindSampler(T_COLOR, flatSampler);

	// VAO
	glBindVertexArray(postProcessTriVao);

	// Draw
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void DeferredRenderer::ShowLightIntensity(const Camera& camera)
{
	ShowTexture(camera, lightIntensityTex);
}

void DeferredRenderer::BindShadowMaps(SceneI& scene)
{
	glActiveTexture(GL_TEXTURE0 + T_SHADOW);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, scene.getSceneLights().getShadowTextureCubemapArray());
	glBindSampler(T_SHADOW, shadowMapSampler);
	glActiveTexture(GL_TEXTURE0 + T_SHADOW_DIR);
	glBindTexture(GL_TEXTURE_2D_ARRAY, scene.getSceneLights().getShadowTextureArrayView());
	glBindSampler(T_SHADOW_DIR, shadowMapSampler);
}

void DeferredRenderer::BindLightBuffers(SceneI& scene)
{
	scene.getSceneLights().BindLightParameters(LU_LIGHT);
	scene.getSceneLights().BindViewProjectionMatrices(LU_LIGHT_MATRIX);
}

void DeferredRenderer::AttachSceneLightIndices(SceneI& scene)
{
	lightAOI.AttachSceneLights(scene.getSceneLights());
}

double DeferredRenderer::ShadowMapTime() const
{
	return shadowMapTime;
}

double DeferredRenderer::DPassTime() const
{
	return dPassTime;
}

double DeferredRenderer::GPassTime() const
{
	return gPassTime;
}

double DeferredRenderer::LPassTime() const
{
	return lPassTime;
}

double DeferredRenderer::MergeTime() const
{
	return mergeTime;
}