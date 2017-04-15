/**

*/

#ifndef __SCENELIGHTS_H__
#define __SCENELIGHTS_H__

#include "IEUtility/IEVector4.h"
#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEBoundingSphere.h"
#include "StructuredBuffer.h"
#include "DrawPoint.h"
#include <cstdint>

struct Camera;
class DrawBuffer;
class GPUBuffer;
class FrameTransformBuffer;

struct Light
{
	IEVector4 position;			// position.w is the light type
	IEVector4 direction;		// direction.w is effecting radius
	IEVector4 color;			// color.a is intensity
};

enum class LightType
{
	POINT = 0,
	DIRECTIONAL = 1,
	SPOT = 2,
	SPHERICAL = 3,
	RECTANGULAR = 4,
	END
};
static constexpr uint32_t LightTypeCount = static_cast<uint32_t>(LightType::END);

struct LightStatus
{
	bool castShadow;
	bool enable;
};

//struct ShadowMapShaders;

class SceneLights
{
	public:
		static constexpr int				CubeSide = 6;

	private:

		// Point Light Shadow Cubemap Related Stuff		
		static const IEVector3				pLightDir[CubeSide];
		static const IEVector3				pLightUp[CubeSide];

		static const IEVector3				aLightDir[CubeSide];
		static const IEVector3				aLightUp[CubeSide];

		// Sparse texture cubemap array
		// One Shadowmap for each light
		// Directional Lights have one side used (others not allocated)
		// Area Lights only use 5 sides of the cube map
		// GPU Buffer
		StructuredBuffer<uint8_t>			gpuBuffer;
		size_t								lightOffset;
		size_t								matrixOffset;

		// Textures and Framebuffers
		GLuint								lightShadowMaps;
		GLuint								shadowMapArrayView;
		GLuint								shadowMapCubeDepth;
		std::vector<GLuint>					shadowMapViews;
		std::vector<GLuint>					shadowMapFBOs;

		// CPU Part
		std::vector<Light>					lights;
		std::vector<IEMatrix4x4>			lightViewProjMatrices;
		std::vector<IEMatrix4x4>			lightProjMatrices;
		std::vector<IEMatrix4x4>			lightInvViewProjMatrices;
		std::vector<bool>					lightShadowCast;

		static float						CalculateCascadeLength(float frustumFar,
																   unsigned int cascadeNo);
		static IEBoundingSphere				CalculateShadowCascasde(float cascadeNear,
																	float cascadeFar,
																	const Camera& camera,
																	const IEVector3& lightDir);
		void								GenerateMatrices(const Camera& camera);
		

	protected:
	public:
		// Constructors & Destructor
											SceneLights();
											SceneLights(const std::vector<Light>& lights);
											SceneLights(const SceneLights&) = delete;
											SceneLights(SceneLights&&);
		SceneLights&						operator=(SceneLights&&);
		SceneLights&						operator=(const SceneLights&) = delete;
											~SceneLights();

		uint32_t							TotalCount() const;
		uint32_t							AreaLightCount() const;
		uint32_t							DirectionalLightCount() const;
		uint32_t							PointLightCount() const;

		void								ChangeLightPos(uint32_t index, IEVector3 position);
		void								ChangeLightDir(uint32_t index, IEVector3 direction);
		void								ChangeLightColor(uint32_t index, IEVector3 color);
		void								ChangeLightRadius(uint32_t index, float radius);
		void								ChangeLightIntensity(uint32_t index, float intensity);
		void								ChangeLightShadow(uint32_t index, bool shadowStatus);

		IEVector3							getLightPos(uint32_t index) const;
		LightType							getLightType(uint32_t index) const;
		IEVector3							getLightDir(uint32_t index) const;
		IEVector3							getLightColor(uint32_t index) const;
		float								getLightRadius(uint32_t index) const;
		float								getLightIntensity(uint32_t index) const;
		bool								getLightCastShadow(uint32_t index) const;
		const std::vector<IEMatrix4x4>&		getLightProjMatrices();
		const std::vector<IEMatrix4x4>&		getLightInvViewProjMatrices();

		// GL States
		void								BindLightFramebuffer(uint32_t light);
		GLuint								BindViewProjectionMatrices(GLuint bindPoint);
		GLuint								BindLightParameters(GLuint bindPoint);

		GLuint								getShadowArrayGL();
		GLuint								getGLBuffer();
		size_t								getLightOffset();
		size_t								getMatrixOffset();
		
		void								SendVPMatricesToGPU();
		void								SendLightDataToGPU();
};

#endif //__SCENE_H__