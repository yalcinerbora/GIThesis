/**

Empty Solution
Just Renders the scene

*/

#ifndef __DEFERREDRENDERER_H__
#define __DEFERREDRENDERER_H__

#include <array>
#include "MeshBatchI.h"
#include "Shader.h"
#include "GBuffer.h"
#include "DrawPoint.h"
#include "StructuredBuffer.h"
#include "FrameTransformBuffer.h"
#include "IEUtility/IEVector3.h"
#include "SceneLights.h"

struct Camera;
class SceneI;
class RectPrism;

struct InvFrameTransform
{
	IEMatrix4x4 invViewProjection;
	IEVector4	camPos;				// Used to generate eye vector
	IEVector4	camDir;				// Used to calculate cascades
	uint32_t	viewport[4];		// Used to generate uv coords from gl_fragCoord
	IEVector4	depthHalfNear;
};

using InvFrameTransformBuffer = StructuredBuffer<InvFrameTransform>;
using LightDrawArray = std::array<DrawPointIndexed, LightTypeCount>;

// Deferred Renderer Light Shape
class LightDrawBuffer
{	
	public:
		static constexpr uint32_t	DirectionalCascadesCount = 4;
		static constexpr uint32_t	ShadowMapMipCount = 8;
		static constexpr uint32_t	ShadowMipSampleCount = 3;
		static constexpr GLsizei	ShadowMapWH = /*512;*/1024;//*2048;*///4096;

	private:
		// Statics
		static constexpr char*		LightAOIFileName = "lightAOI.gfg";

		// Buffer Data
		LightDrawArray				lightDrawParams;
		std::vector<uint32_t>		lightIndexBuffer;

		// Buffer Storage order
		// 1- Draw Param (Static Depends on #LightTypes)
		// 2- AOI Vertex (Static Depends on #LightTypes)
		// 3- AOI Vertex Index (Static Depends on #LightTypes)
		// 4- Light index buffer(Dynamic Depends on #Lights)
		StructuredBuffer<uint8_t>	gpuBuffer;

		// Light AOI VAO
		GLuint						lightVAO;

	public:
		// Constructors & Destructor
									LightDrawBuffer();
									LightDrawBuffer(const LightDrawBuffer&) = delete;
		LightDrawBuffer&			operator=(const LightDrawBuffer&) = delete;
									~LightDrawBuffer();

		void						ChangeLightCounts();

		void						BindVAO();
		void						BindDrawIndirectBuffer();		
		void						DrawCall();

};

static_assert(LightDrawBuffer::DirectionalCascadesCount <= SceneLights::CubeSide, "At most 6 cascades can be created");

class DeferredRenderer
{
	public:
		// Geometry Buffer Dimensions
		static constexpr GLsizei	GBuffWidth = /*160;*//*320;*//*640;*//*800;*/1280;/*1600;*///*1920;*//*2560;*///3840;
		static constexpr GLsizei	GBuffHeight = /*90;*//*180;*//*360;*//*450;*/720;/*900;*///*1080;*//*1440;*///2160;;
	
		

	private:
		static constexpr float		postProcessTriData[6] =
		{
			3.0f, -1.0f,
			-1.0f, 3.0f,
			-1.0f, -1.0f
		};

		// Geom Buffer Write Shaders
		Shader						vertGBufferSkeletal;
		Shader						vertGBufferWrite;
		Shader						fragGBufferWrite;

		// Depth Prepass Shaders
		Shader						vertDPass;
		Shader						vertDPassSkeletal;

		// Light Pass Shaders
		Shader						vertLightPass;
		Shader						fragLightPass;

		// Post Process Shaders
		Shader						vertPPGeneric;
		Shader						fragLightApply;
		Shader						fragPPGeneric;
		Shader						fragPPNormal;
		Shader						fragPPDepth;

		// Shader for shadowmap
		Shader						vertShadowMap;
		Shader						vertShadowMapSkeletal;
		Shader						geomAreaShadowMap;
		Shader						geomPointShadowMap;
		Shader						geomDirShadowMap;
		Shader						fragShadowMap;
		Shader						computeHierZ;

		// Geometry Buffer
		GBuffer						gBuffer;

		// Light AOI Buffer			
		LightDrawBuffer				lightAOI;

		// Frame Transform
		FrameTransformBufferData	fTransform;
		InvFrameTransform			ifTransform;
		StructuredBuffer<uint8_t>	transformBuffer;

		// Light Intensity Texture and sRGB output texture
		GLuint						lightIntensityTex;
		GLuint						lightIntensityFBO;
		GLuint						sRGBEndTex;
		GLuint						sRGBEndFBO;

		// Post Process Triangle 
		GLuint						postProcessTriVao;
		GLuint						postProcessTriBuffer;

		// Samplers
		GLuint						flatSampler;
		GLuint						linearSampler;
		GLuint						shadowMapSampler;
		
		void						BindShadowMapGeometryShader(LightType t);
		void						BindShadowMapVertexShader(MeshBatchType t);
		
		void						BindInvFrameTransform(GLuint bindingPoint);
		void						BindFrameTransform(GLuint bindingPoint);
		void						UpdateFTransformBuffer();
		void						UpdateInvFTransformBuffer();

	protected:

	public:
		// Constructors & Destructor
									DeferredRenderer();
									DeferredRenderer(const DeferredRenderer&) = delete;
		DeferredRenderer&			operator=(const DeferredRenderer&) = delete;
									~DeferredRenderer();

		GBuffer&					getGBuffer();
		GLuint						getLightIntensityBufferGL();
//		InvFrameTransformBuffer&	GetInvFTransfrom();
//		FrameTransformBuffer&		GetFTransform();

		void						RefreshInvFTransform(const Camera&,
														 GLsizei width,
														 GLsizei height);

		void						Render(SceneI&, const Camera&, bool directLight, const IEVector3& ambientColor);
		void						PopulateGBuffer(SceneI&, const Camera&);

		// Do stuff by function
		void						GenerateShadowMaps(SceneI&, const Camera&);
		void						DPass(SceneI&, const Camera&);
		void						GPass(SceneI&, const Camera&);
		void						ClearLI(const IEVector3& ambientColor);
		void						LightPass(SceneI&, const Camera&);
		void						Present(const Camera&);

		// Directly Renders Buffers
		void						ShowColorGBuffer(const Camera& camera);
		void						ShowNormalGBuffer(const Camera& camera);
		void						ShowDepthGBuffer(const Camera& camera);
		void						ShowLIBuffer(const Camera& camera);
		void						ShowTexture(const Camera& camera, GLuint tex, RenderTargetLocation location = RenderTargetLocation::COLOR);
		
		void						AddToLITexture(GLuint texture);

		void						BindShadowMaps(SceneI&);
		void						BindLightBuffers(SceneI&);
};
#endif //__DEFERREDRENDERER_H__