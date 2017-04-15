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

	glGenBuffers(1, &lightShapeBuffer);
	glBindBuffer(GL_COPY_WRITE_BUFFER, lightShapeBuffer);
	glBufferData(GL_COPY_WRITE_BUFFER, vData.size(), vData.data(), GL_STATIC_DRAW);

	glGenBuffers(1, &lightShapeIndexBuffer);
	glBindBuffer(GL_COPY_WRITE_BUFFER, lightShapeIndexBuffer);
	glBufferData(GL_COPY_WRITE_BUFFER, viData.size(), viData.data(), GL_STATIC_DRAW);

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

	// Create VAO
	// PostProcess VAO
	glGenVertexArrays(1, &lightVAO);
	glBindVertexArray(lightVAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lightShapeIndexBuffer);

	// Pos
	glBindVertexBuffer(0, lightShapeBuffer, 0, sizeof(float) * 3);
	glEnableVertexAttribArray(IN_POS);
	glVertexAttribFormat(IN_POS, 3, GL_FLOAT, false, 0);
	glVertexAttribBinding(IN_POS, 0);

	// Index
	glBindVertexBuffer(1, lightIndexBuffer.getGLBuffer(), 0, sizeof(uint32_t));
	glVertexBindingDivisor(1, 1);
	glEnableVertexAttribArray(IN_LIGHT_INDEX);
	glVertexAttribIFormat(IN_LIGHT_INDEX, 1, GL_UNSIGNED_INT, 0);
	glVertexAttribBinding(IN_LIGHT_INDEX, 1);

	assert(lightsGPU.CPUData().size() == lightShadowCast.size());

}

void LightDrawBuffer::ChangeLightCounts(const std::vector<Light>& lights)
{

	//// Draw Buffers
	//lightDrawParams.AddData(drawParamsGeneric[static_cast<int>(LightType::POINT)]);
	//lightDrawParams.AddData(drawParamsGeneric[static_cast<int>(LightType::DIRECTIONAL)]);
	//lightDrawParams.AddData(drawParamsGeneric[static_cast<int>(LightType::RECTANGULAR)]);

	//lightDrawParams.CPUData()[static_cast<int>(LightType::POINT)].instanceCount = pCount;
	//lightDrawParams.CPUData()[static_cast<int>(LightType::DIRECTIONAL)].instanceCount = dCount;
	//lightDrawParams.CPUData()[static_cast<int>(LightType::RECTANGULAR)].instanceCount = aCount;
	//lightDrawParams.CPUData()[static_cast<int>(LightType::POINT)].baseInstance = 0;
	//lightDrawParams.CPUData()[static_cast<int>(LightType::DIRECTIONAL)].baseInstance = pCount;
	//lightDrawParams.CPUData()[static_cast<int>(LightType::RECTANGULAR)].baseInstance = pCount + dCount;
	//lightDrawParams.SendData();


	// Light Draw Param Generation
	uint32_t dCount = 0, aCount = 0, pCount = 0, i = 0;
	uint32_t dIndex = 0, aIndex = 0, pIndex = 0;
	std::vector<uint32_t>& lIndexBuff = lightIndexBuffer.CPUData();
	lIndexBuff.resize(3);
	for(const Light& l : lights)
	{
		if(ParseLightType(l.position.getW()) == LightType::RECTANGULAR)
			aCount++;
		else if(ParseLightType(l.position.getW()) == LightType::DIRECTIONAL)
			dCount++;
		else if(ParseLightType(l.position.getW()) == LightType::POINT)
			pCount++;
	}
	for(const Light& l : lightsGPU.CPUData())
	{
		if(l.position.getW() == static_cast<float>(static_cast<int>(LightType::AREA)))
		{
			lIndexBuff[pCount + dCount + aIndex] = i;
			aIndex++;
		}
		else if(l.position.getW() == static_cast<float>(static_cast<int>(LightType::DIRECTIONAL)))
		{
			lIndexBuff[pCount + dIndex] = i;
			dIndex++;
		}
		else if(l.position.getW() == static_cast<float>(static_cast<int>(LightType::POINT)))
		{
			lIndexBuff[0 + pIndex] = i;
			pIndex++;
		}
		i++;
	}
	lightIndexBuffer.SendData();

}

void LightDrawBuffer::BindVAO()
{
	glBindVertexArray(lightVAO);
}

void LightDrawBuffer::BindDrawIndirectBuffer()
{
	gpuBuffer.BindAsDrawIndirectBuffer();
}

void LightDrawBuffer::DrawCall()
{
	GLsizei offset = static_cast<GLsizei>(indexOffset);
	glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 
								(void*)(offset), LightTypeCount, sizeof(DrawPointIndexed));
}

DeferredRenderer::DeferredRenderer()
	// Geom Write
	: vertGBufferSkeletal(ShaderType::VERTEX, "Shaders/GWriteSkeletal.vert")
	, vertGBufferWrite(ShaderType::VERTEX, "Shaders/GWriteGeneric.vert")
	, fragGBufferWrite(ShaderType::FRAGMENT, "Shaders/GWriteGeneric.frag")
	// Depth Prepass
	, vertDPass(ShaderType::VERTEX, "Shaders/DPass.vert")
	, vertDPassSkeletal(ShaderType::VERTEX, "Shaders/DPassSkeletal.vert")
	// Light Pass
	, vertLightPass(ShaderType::VERTEX, "Shaders/LightPass.vert")
	, fragLightPass(ShaderType::FRAGMENT, "Shaders/LightPass.frag")
	// Post process
	, vertPPGeneric(ShaderType::VERTEX, "Shaders/PProcessGeneric.vert")
	, fragLightApply(ShaderType::FRAGMENT, "Shaders/PPLightPresent.frag")
	, fragPPGeneric(ShaderType::FRAGMENT, "Shaders/PProcessGeneric.frag")
	, fragPPNormal(ShaderType::FRAGMENT, "Shaders/PProcessNormal.frag")
	, fragPPDepth(ShaderType::FRAGMENT, "Shaders/PProcessDepth.frag")
	// Shadow Maps
	, vertShadowMap(ShaderType::VERTEX, "Shaders/ShadowMap.vert")
	, vertShadowMapSkeletal(ShaderType::VERTEX, "Shaders/ShadowMapSkeletal.vert")
	, geomAreaShadowMap(ShaderType::GEOMETRY, "Shaders/ShadowMapA.geom")
	, geomDirShadowMap(ShaderType::GEOMETRY, "Shaders/ShadowMapD.geom")
	, geomPointShadowMap(ShaderType::GEOMETRY, "Shaders/ShadowMapP.geom")
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
	, lightIntensityTex(0)
	, lightIntensityFBO(0)
	, sRGBEndTex(0)
	, sRGBEndFBO(0)
{
	// Light Intensity Tex
	glGenTextures(1, &lightIntensityTex);
	glGenFramebuffers(1, &lightIntensityFBO);

	glBindTexture(GL_TEXTURE_2D, lightIntensityTex);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, GBuffWidth, GBuffHeight);

	glBindFramebuffer(GL_FRAMEBUFFER, lightIntensityFBO);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, lightIntensityTex, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, gBuffer.getDepthGL(), 0);
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

	// SRGB Tex
	glGenTextures(1, &sRGBEndTex);
	glGenFramebuffers(1, &sRGBEndFBO);
	glBindTexture(GL_TEXTURE_2D, sRGBEndTex);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_SRGB8_ALPHA8, gBuffWidth, gBuffHeight);

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
	
	// PostProcess VAO
	glGenBuffers(1, &postProcessTriBuffer);
	glGenVertexArrays(1, &postProcessTriVao);

	glBindBuffer(GL_COPY_WRITE_BUFFER, postProcessTriBuffer);
	glBufferData(GL_COPY_WRITE_BUFFER, sizeof(float) * 6, postProcessTriData, GL_STATIC_DRAW);

	glBindVertexArray(postProcessTriVao);
	glBindVertexBuffer(0, postProcessTriBuffer, 0, sizeof(float) * 2);
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
	glDeleteTextures(1, &sRGBEndTex);
	glDeleteFramebuffers(1, &lightIntensityFBO);
	glDeleteFramebuffers(1, &sRGBEndFBO);
	glDeleteBuffers(1, &postProcessTriBuffer);
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
//
//InvFrameTransformBuffer& DeferredRenderer::GetInvFTransfrom()
//{
//	return invFrameTransform;
//}
//
//FrameTransformBuffer& DeferredRenderer::GetFTransform()
//{
//	return cameraTransform;
//}

void DeferredRenderer::GenerateShadowMaps(SceneI& scene, const Camera& camera)
{
	fragShadowMap.Bind();
	uint32_t lightCount = static_cast<unsigned int>(scene.getSceneLights().lightsGPU.CPUData().size());

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

	// Binding
	//cameraTransform.Bind();
	scene.getSceneLights().lightViewProjMatrices.BindAsShaderStorageBuffer(LU_LIGHT_MATRIX);

	// Render Loop
	for(unsigned int i = 0; i < lightCount; i++)
	{
		// FBO Bind and render calls
		glBindFramebuffer(GL_FRAMEBUFFER, scene.getSceneLights().shadowMapFBOs[i]);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if(!scene.getSceneLights().lightShadowCast[i])
			continue;

		// Draw Batches
		const std::vector<MeshBatchI*>& batches = scene.getBatches();
		for(size_t j = 0; j < batches.size(); j++)
		{
			if(batches[j]->MeshType() == MeshBatchType::SKELETAL)
			{
				vertShadowMapSkeletal.Bind();
				auto batchPtr = static_cast<MeshBatchSkeletal*>(batches[j]);
				batchPtr->getJointTransforms().BindAsShaderStorageBuffer(LU_JOINT_TRANS);
			}
			else
			{
				vertShadowMap.Bind();
			}

			VertexBuffer& currentVertexBuffer = batches[j]->getVertexBuffer();
			DrawBuffer& currentDrawBuffer = batches[j]->getDrawBuffer();

			currentVertexBuffer.Bind();
			currentDrawBuffer.BindAsDrawIndirectBuffer();

			// Base poly offset gives ok results on point-area lights
			glPolygonOffset(2.64f, 512.0f);

			const Light& currentLight = scene.getSceneLights().lightsGPU.CPUData()[i];
			LightType t = static_cast<LightType>(static_cast<uint32_t>(currentLight.position.getW()));
			switch(t)
			{
				case LightType::POINT: geomPointShadowMap.Bind(); break;
				case LightType::DIRECTIONAL: geomDirShadowMap.Bind();
					glPolygonOffset(6.12f, 1024.0f); // Higher offset req since camera span is large
					break;
				case LightType::RECTANGULAR: geomAreaShadowMap.Bind(); break;
			}
			glUniform1ui(U_LIGHT_ID, static_cast<GLuint>(i));

			currentDrawBuffer.BindModelTransform(LU_MTRANSFORM);
			
			// Draw Call
			currentDrawBuffer.DrawCallMulti();
			// Stays Here for Debugging purposes (nsight states)
			// currentDrawBuffer.DrawCallMultiState();
		}
	}

	glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT);
	glFlush();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Hierarchical z-buffers
	GLuint lightTex = scene.getSceneLights().shadowMapArrayView;
	computeHierZ.Bind();
	for(unsigned int i = 0; i < SceneLights::shadowMipCount - 1; i++)
	{
		// Reduce Entire Level at once
		// TODO:
		GLuint depthSize = SceneLights::shadowMapWH >> (i + 1);
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
}

void DeferredRenderer::GPass(SceneI& scene, const Camera& camera)
{
	gBuffer.BindAsFBO();
	gBuffer.AlignViewport();

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);
	glDepthMask(false);
	glDepthFunc(GL_EQUAL);
	glColorMask(true, true, true, true);
	glClear(GL_COLOR_BUFFER_BIT);

	// Without Depth Prepass
	glDepthMask(true);
	glClear(GL_DEPTH_BUFFER_BIT);
	glDepthFunc(GL_LEQUAL);
	
	// Camera Transform
	fTransform = camera.GenerateTransform();
	UpdateFTransformBuffer();
	BindFrameTransform(U_FTRANSFORM);

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	fragGBufferWrite.Bind();

	// DrawCall
	// Draw Batches
	const std::vector<MeshBatchI*>& batches = scene.getBatches();
	for(unsigned int i = 0; i < batches.size(); i++)
	{
		if(batches[i]->MeshType() == MeshBatchType::SKELETAL)
		{
			vertGBufferSkeletal.Bind();
			MeshBatchSkeletal* batchPtr = static_cast<MeshBatchSkeletal*>(batches[i]);
			batchPtr->getJointTransforms().BindAsShaderStorageBuffer(LU_JOINT_TRANS);
		}
		else
		{
			vertGBufferWrite.Bind();
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
}

void DeferredRenderer::ClearLI(const IEVector3& ambientColor)
{
	glBindFramebuffer(GL_FRAMEBUFFER, lightIntensityFBO);
	glClearColor(ambientColor.getX(), ambientColor.getY(), ambientColor.getZ(), 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void DeferredRenderer::LightPass(SceneI& scene, const Camera& camera)
{
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
	RefreshInvFTransform(camera, gBuffWidth, gBuffHeight);
	BindInvFrameTransform(U_INVFTRANSFORM);
	BindFrameTransform(U_FTRANSFORM);
	

	// Bind LightIntensity Buffer as framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, lightIntensityFBO);
	glViewport(0, 0, gBuffWidth, gBuffHeight);

	// Shader and Shader Uniform Binds
	Shader::Unbind(ShaderType::GEOMETRY);
	vertLightPass.Bind();
	fragLightPass.Bind();
	glUniform1ui(U_SHADOW_MIP_COUNT, static_cast<GLuint>(SceneLights::mipSampleCount));
	glUniform1ui(U_SHADOW_MAP_WH, static_cast<GLuint>(SceneLights::shadowMapWH));

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
}

void DeferredRenderer::DPass(SceneI& scene, const Camera& camera)
{
	// Depth Pass that to utilize early fragment test
	gBuffer.BindAsFBO();
	gBuffer.AlignViewport();

	// States
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);
	glDepthMask(true);
	glColorMask(false, false, false, false);
	glDepthFunc(GL_LESS);
	glClear(GL_DEPTH_BUFFER_BIT |
			GL_STENCIL_BUFFER_BIT);
	
	// Shaders
	fragShadowMap.Bind();
	Shader::Unbind(ShaderType::GEOMETRY);

	// Uniform Buffers
	BindFrameTransform(U_FTRANSFORM);
	
	// Draw Batches
	const std::vector<MeshBatchI*>& batches = scene.getBatches();
	for(unsigned int i = 0; i < batches.size(); i++)
	{
		if(batches[i]->MeshType() == MeshBatchType::SKELETAL)
		{
			vertDPassSkeletal.Bind();
			MeshBatchSkeletal* batchPtr = static_cast<MeshBatchSkeletal*>(batches[i]);
			batchPtr->getJointTransforms().BindAsShaderStorageBuffer(LU_JOINT_TRANS);
		}
		else
		{
			vertDPass.Bind();
		}

		VertexBuffer& currentVertexBuffer = batches[i]->getVertexBuffer();
		DrawBuffer& currentDrawBuffer = batches[i]->getDrawBuffer();

		currentVertexBuffer.Bind();
		currentDrawBuffer.BindAsDrawIndirectBuffer();
		currentDrawBuffer.BindModelTransform(LU_MTRANSFORM);

		currentDrawBuffer.DrawCallMulti();
		// Stays Here for Debugging purposes (nsight states)
		//currentDrawBuffer.DrawCallMultiState();
	}
}

void DeferredRenderer::Present(const Camera& camera)
{
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
			   static_cast<GLsizei>(gBuffWidth),
			   static_cast<GLsizei>(gBuffHeight));

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
}

void DeferredRenderer::RefreshInvFTransform(const Camera& camera,
											GLsizei width,
											GLsizei height)
{
	FrameTransformBufferData ft = camera.GenerateTransform();

	float depthRange[2];
	glGetFloatv(GL_DEPTH_RANGE, depthRange);
	ifTransform = InvFrameTransform
	{
		ft.view.Inverse() * ft.projection.Inverse(),
		IEVector4(camera.pos, CalculateCascadeLength(camera.far, 0.0f)),
		IEVector4((camera.centerOfInterest - camera.pos).NormalizeSelf()),
		{0, 0, static_cast<unsigned int>(width), static_cast<unsigned int>(height)},
		{depthRange[0], depthRange[1], 0.0f, 0.0f}
	};
	UpdateInvFTransformBuffer();
}

void DeferredRenderer::PopulateGBuffer(SceneI& scene, const Camera& camera)
{
	// Depth Pre-Pass
//	DPass(scene, camera);

	// Actual Render
	// G Pass
	GPass(scene, camera);
}

void DeferredRenderer::Render(SceneI& scene, const Camera& camera, bool directLight,
							  const IEVector3& ambientColor)
{
	// Camera Transform Update
	fTransform = camera.GenerateTransform();
	UpdateFTransformBuffer();

	// Shadow Map Generation
	GenerateShadowMaps(scene, camera);

	// GPass
	PopulateGBuffer(scene, camera);
	
	// Clear LI with ambient color
	ClearLI(ambientColor);

	// Light Pass
	if(directLight)
	{
		LightPass(scene, camera);
	}

	// Light Intensity Merge
	Present(camera);

	// All Done!
}

void DeferredRenderer::ShowTexture(const Camera& camera, GLuint tex, RenderTargetLocation location)
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

	if(location == RenderTargetLocation::COLOR)
		fragPPGeneric.Bind();
	else if(location == RenderTargetLocation::NORMAL)
	{
		fragPPNormal.Bind();
	}
	else if(location == RenderTargetLocation::DEPTH)
	{
		fragPPDepth.Bind();
		glUniform2f(U_NEAR_FAR, camera.near, camera.far);
	}
	
	// Texture
	glActiveTexture(GL_TEXTURE0 + T_COLOR);
	glBindTexture(GL_TEXTURE_2D, tex);
	glBindSampler(T_COLOR, linearSampler);

	// VAO
	glBindVertexArray(postProcessTriVao);

	// Draw
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void DeferredRenderer::ShowColorGBuffer(const Camera& camera)
{
	ShowTexture(camera, gBuffer.getColorGL());
}

void DeferredRenderer::ShowNormalGBuffer(const Camera& camera)
{
	ShowTexture(camera, gBuffer.getNormalGL(), RenderTargetLocation::NORMAL);
}

void DeferredRenderer::ShowDepthGBuffer(const Camera& camera)
{
	ShowTexture(camera, gBuffer.getDepthGL(), RenderTargetLocation::DEPTH);
}

void DeferredRenderer::ShowLIBuffer(const Camera& camera)
{
	ShowTexture(camera, lightIntensityTex);
}

void DeferredRenderer::BindShadowMaps(SceneI& scene)
{
	glActiveTexture(GL_TEXTURE0 + T_SHADOW);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, scene.getSceneLights().lightShadowMaps);
	glBindSampler(T_SHADOW, shadowMapSampler);
	glActiveTexture(GL_TEXTURE0 + T_SHADOW_DIR);
	glBindTexture(GL_TEXTURE_2D_ARRAY, scene.getSceneLights().shadowMapArrayView);
	glBindSampler(T_SHADOW_DIR, shadowMapSampler);
}

void DeferredRenderer::BindLightBuffers(SceneI& scene)
{
	scene.getSceneLights().lightsGPU.BindAsShaderStorageBuffer(LU_LIGHT);
	scene.getSceneLights().lightViewProjMatrices.BindAsShaderStorageBuffer(LU_LIGHT_MATRIX);
}