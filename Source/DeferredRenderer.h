/**

Empty Solution
Just Renders the scene

*/

#ifndef __DEFERREDRENDERER_H__
#define __DEFERREDRENDERER_H__

#include "Shader.h"
#include "FrameTransformBuffer.h"
#include "GBuffer.h"
#include "DrawPoint.h"
#include "StructuredBuffer.h"

struct Camera;
class SceneI;
class RectPrism;

struct InvFrameTransform
{
	IEMatrix4x4 invView;
	IEMatrix4x4 invProjection;
	IEMatrix4x4 invViewRotation;
	IEVector4 camPos;		// Used to generate eye vector
	uint32_t viewport[4];	// Used to generate uv coords from gl_fragCoord
};

using InvFrameTransformBuffer = StructuredBuffer<InvFrameTransform>;

class DeferredRenderer
{
	private:
		static const GLsizei	gBuffWidth;
		static const GLsizei	gBuffHeight;

		static const float		postProcessTriData[6];

		Shader					vertexGBufferWrite;
		Shader					fragmentGBufferWrite;
		Shader					vertDPass;

		Shader					vertLightPass;
		Shader					fragLightPass;

		Shader					vertPPGeneric;
		Shader					fragLightApply;

		// Shader for shadowmap
		Shader					fragShadowMap;
		Shader					vertShadowMap;

		Shader					geomAreaShadowMap;
		Shader					geomPointShadowMap;
		Shader					geomDirShadowMap;

		GBuffer					gBuffer;
		FrameTransformBuffer	cameraTransform;
		InvFrameTransformBuffer invFrameTransform;

		// Light Object Meshes (vertex & index buffers)
		// Light Object VAO's
		GLuint					lightIntensityTex;
		GLuint					lightIntensityFBO;

		GLuint					postProcessTriVao;
		GLuint					postProcessTriBuffer;

		GLuint					flatSampler;
		GLuint					linearSampler;
		GLuint					shadowMapSampler;

	protected:
		void					GenerateShadowMaps(SceneI&, const Camera&, const RectPrism& viewFrustum);
		void					DPass(SceneI&, const Camera&);
		void					GPass(SceneI&, const Camera&);
		void					LightPass(SceneI&, const Camera&);
		void					LightMerge(const Camera&);

	public:
								DeferredRenderer();
								DeferredRenderer(const DeferredRenderer&) = delete;
		DeferredRenderer&		operator=(const DeferredRenderer&) = delete;
								~DeferredRenderer();

		GBuffer&				GetGBuffer();
		void					Render(SceneI&, const Camera&);
};
#endif //__DEFERREDRENDERER_H__