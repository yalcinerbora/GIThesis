/**

*/

#ifndef __SCENELIGHTS_H__
#define __SCENELIGHTS_H__

#include "IEUtility/IEVector4.h"
#include "IEUtility/IEMatrix4x4.h"
#include "StructuredBuffer.h"
#include "ArrayStruct.h"
#include "DrawPoint.h"
#include <cstdint>

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
	AREA = 2
};

struct LightStatus
{
	bool castShadow;
	bool enable;
};

//struct ShadowMapShaders;

class SceneLights
{
	private:
		friend class DeferredRenderer;

		static const IEVector3	pLightDir[6];
		static const IEVector3	pLightUp[6];

		static const IEVector3	aLightDir[6];
		static const IEVector3	aLightUp[6];	

		// Sparse texture cubemap array
		// One Shadowmap for each light
		// Directional Lights have one side used (others not allocated)
		// Area Lights only use 5 sides of the cube map
		StructuredBuffer<Light>				lightsGPU;
		StructuredBuffer<IEMatrix4x4>		lightViewProjMatrices;
		GLuint								lightShadowMaps;
		GLuint								shadowMapArrayView;
		GLuint								shadowMapCubeDepth;
		std::vector<GLuint>					shadowMapViews;
		std::vector<GLuint>					shadowMapFBOs;
		std::vector<bool>					lightShadowCast;

		// Light Shape Related
		static const char*					lightAOIFileName;
		static GLuint						lightShapeBuffer;
		static GLuint						lightShapeIndexBuffer;
		static DrawPointIndexed				drawParamsGeneric[3];	// Only Instance Count is not used

		StructuredBuffer<DrawPointIndexed>	lightDrawParams;
		GLuint								lightVAO;
		StructuredBuffer<uint32_t>			lightIndexBuffer;

	protected:
	public:
		// Constructors & Destructor
									SceneLights(const Array32<Light>& lights);
									SceneLights(const SceneLights&) = delete;
		SceneLights&				operator=(const SceneLights&) = delete;
									~SceneLights();

		uint32_t					Count() const;
		GLuint						GetLightBufferGL();
		GLuint						GetShadowArrayGL();
		GLuint						GetVPMatrixGL();
		
		void						ChangeLightPos(uint32_t index, IEVector3 position);
		void						ChangeLightDir(uint32_t index, IEVector3 direction);
		void						ChangeLightColor(uint32_t index, IEVector3 color);
		void						ChangeLightRadius(uint32_t index, float radius);
		void						ChangeLightIntensity(uint32_t index, float intensity);
		void						ChangeLightShadow(uint32_t index, bool shadowStatus);

		IEVector3					GetLightPos(uint32_t index) const;
		LightType					GetLightType(uint32_t index) const;
		IEVector3					GetLightDir(uint32_t index) const;
		IEVector3					GetLightColor(uint32_t index) const;
		float						GetLightRadius(uint32_t index) const;
		float						GetLightIntensity(uint32_t index) const;
		bool						GetLightShadow(uint32_t index) const;

		static const GLsizei		shadowMapWH;
		static const uint32_t		numShadowCascades;
		static const uint32_t		shadowMipCount;
		static const uint32_t		mipSampleCount;
};

#endif //__SCENE_H__