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
#include "IEUtility/IEVector3.h"

struct Camera;
class SceneI;
class RectPrism;

struct BoundingSphere
{
	IEVector3 center;
	float radius;
};

struct InvFrameTransform
{
	IEMatrix4x4 invViewProjection;
	IEVector4 camPos;		// Used to generate eye vector
	IEVector4 camDir;		// Used to calculate cascades
	uint32_t viewport[4];	// Used to generate uv coords from gl_fragCoord
	IEVector4 depthHalfNear;
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

		static BoundingSphere	CalculateShadowCascasde(float cascadeNear,
														float cascadeFar,
														const Camera& camera,
														const IEVector3& lightDir);
		static float			CalculateCascadeLength(float frustumFar);

	protected:
		void					GenerateShadowMaps(SceneI&, const Camera&);
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