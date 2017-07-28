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
#include "IEUtility/IEVector3.h"
#include "SceneLights.h"
#include "Globals.h"
#include "OGLTimer.h"

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
using LightShaderArray = std::array<Shader, LightTypeCount>;
using MeshBatchShaderArray = std::array<Shader, MeshBatchTypeCount>;

// Deferred Renderer Light Shape
class LightDrawBuffer
{	
	public:
		static constexpr uint32_t	DirectionalCascadesCount = 6;
		static constexpr uint32_t	ShadowMipSampleCount = 3;
		static constexpr GLsizei	ShadowMapWH = /*512;*//*1024;*/2048;//4096;
		static constexpr uint32_t	ShadowMapMipCount = 4;

	private:
		// Statics
		static constexpr char*		LightAOIFileName = "lightAOI.gfg";

		// Buffer Data
		LightDrawArray				lightDrawParams;

		// Buffer Storage order
		// 1- Draw Param (Static Depends on #LightTypes)
		// 2- AOI Vertex (Static Depends on #LightTypes)
		// 3- AOI Vertex Index (Static Depends on #LightTypes)
		StructuredBuffer<uint8_t>	gpuData;
		size_t						drawOffset;
		size_t						vertexOffset;
		size_t						indexOffset;

		// Light AOI VAO
		GLuint						lightVAO;

	public:
		// Constructors & Destructor
									LightDrawBuffer();
									LightDrawBuffer(const LightDrawBuffer&) = delete;
		LightDrawBuffer&			operator=(const LightDrawBuffer&) = delete;
									~LightDrawBuffer();

		void						AttachSceneLights(SceneLights&);

		void						BindVAO();
		void						BindDrawIndirectBuffer();		
		void						DrawCall();

};

static_assert(LightDrawBuffer::DirectionalCascadesCount <= SceneLights::CubeSide, "At most 6 cascades can be created");

class DeferredRenderer
{
	public:
		// Geometry Buffer Dimensions
		static constexpr GLsizei	GBuffWidth = /*160;*//*320;*//*640;*//*800;*//*1280;*//*1600;*/1920;/*2560;*///3840;
		static constexpr GLsizei	GBuffHeight = /*90;*//*180;*//*360;*//*450;*//*720;*//*900;*/1080;/*1440;*///2160;

	private:
		static constexpr float		postProcessTriData[6] =
		{
			3.0f, -1.0f,
			-1.0f, 3.0f,
			-1.0f, -1.0f
		};

		// Geom Buffer Write Shaders
		MeshBatchShaderArray		vertGBufferWrite;
		Shader						fragGBufferWrite;

		// Depth Prepass Shaders
		MeshBatchShaderArray		vertDPass;
		Shader						fragDPass;

		// Light Pass Shaders
		Shader						vertLightPass;
		Shader						fragLightPass;

		// Post Process Shaders
		Shader						vertPPGeneric;
		Shader						fragLightApply;
		Shader						fragPPGeneric;
		Shader						fragPPGBuffer;
		Shader						fragPPShadowMap;

		// Shader for shadowmap
		MeshBatchShaderArray		vertShadowMap;
		LightShaderArray			geomShadowMap;
		Shader						fragShadowMap;
		Shader						computeHierZ;

		// Geometry Buffer
		GBuffer						gBuffer;

		// Light AOI Buffer			
		LightDrawBuffer				lightAOI;

		// Frame Transform
		FrameTransformData			fTransform;
		InvFrameTransform			ifTransform;
		StructuredBuffer<uint8_t>	gpuData;
		size_t						postTriOffset;
		size_t						fOffset;
		size_t						iOffset;

		// Light Intensity Texture and sRGB output texture
		GLuint						lightIntensityTex;
		GLuint						lightIntensityFBO;
		GLuint						sRGBEndTex;
		GLuint						sRGBEndFBO;

		// Post Process Triangle 
		GLuint						postProcessTriVao;

		// Timings
		OGLTimer					oglTimer;
		double						shadowMapTime;
		double						dPassTime;
		double						gPassTime;		
		double						lPassTime;
		double						mergeTime;

		// Samplers
		GLuint						flatSampler;
		GLuint						linearSampler;
		GLuint						shadowMapSampler;
				
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
		const GBuffer&				getGBuffer() const;
		GLuint						getLightIntensityBufferGL();

		void						RefreshFTransform(const Camera&);
		void						RefreshInvFTransform(SceneI&,
														 const Camera&,
														 GLsizei width,
														 GLsizei height);
		void						BindInvFrameTransform(GLuint bindingPoint) const;
		void						BindFrameTransform(GLuint bindingPoint) const;

		void						Render(SceneI&, const Camera&, bool directLight, 
										   const IEVector3& ambientColor,
										   bool doTiming);		
		void						Present(const Camera&, bool doTiming);


		// Render Calls Individual
		void						GenerateShadowMaps(SceneI&, const Camera&, bool doTiming);
		void						DPass(SceneI&, const Camera&, bool doTiming);
		void						GPass(SceneI&, const Camera&, bool doTiming);
		void						PopulateGBuffer(SceneI&, const Camera&, bool doTiming);
		void						ClearLI(const IEVector3& ambientColor);
		void						LightPass(SceneI&, const Camera&, bool doTiming);

		// Directly Renders Buffers
		void						ShowGBufferTexture(const Camera& camera, RenderScheme);
		void						ShowLightIntensity(const Camera& camera);
		void						ShowShadowMap(const Camera& camera, 
												  SceneI& scene, int lightId, int layer);
		void						ShowTexture(const Camera& camera, GLuint tex);
		
		void						BindShadowMaps(SceneI&);
		void						BindLightBuffers(SceneI&);
		void						AttachSceneLightIndices(SceneI&);

		double						ShadowMapTime() const;
		double						DPassTime() const;
		double						GPassTime() const;
		double						LPassTime() const;
		double						MergeTime() const;
};
#endif //__DEFERREDRENDERER_H__