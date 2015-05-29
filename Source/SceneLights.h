/**

*/

#ifndef __SCENELIGHTS_H__
#define __SCENELIGHTS_H__

#include "IEUtility/IEVector4.h"
#include "IEUtility/IEMatrix4x4.h"
#include "StructuredBuffer.h"
#include "ArrayStruct.h"
#include "Shader.h"
#include "RectPrism.h"
#include <cstdint>

class DrawBuffer;
class GPUBuffer;
class FrameTransformBuffer;

struct Light
{
	IEVector4 position;			// position.w is the light type
	IEVector4 direction;		// direction.w is empty
	IEVector4 color;			// color.a is effecting radius
};

enum class LightType
{
	POINT = 0,
	DIRECTIONAL = 1,
	AREA = 2
};

class SceneLights
{
	private:
		static const GLsizei	shadowMapW;
		static const GLsizei	shadowMapH;

		static const IEVector3	pLightDir[6];
		static const IEVector3	pLightUp[6];

		static const IEVector3	aLightDir[6];
		static const IEVector3	aLightUp[6];

		// Sparse texture cubemap array
		// One Shadowmap for each light
		// Directional Lights have one side used (others not allocated)
		// Area Lights only use 5 sides of the cube map
		StructuredBuffer<Light>			lightsGPU;
		StructuredBuffer<IEMatrix4x4>	viewMatrices;
		GLuint							lightShadowMaps;
		std::vector<GLuint>				shadowMapViews;
		std::vector<GLuint>				shadowMapFBOs;

		// Shader for lightmap
		Shader					fragShadowMap;
		Shader					vertShadowMap;
		Shader					geomAreaShadowMap;
		Shader					geomPointShadowMap;
		Shader					geomDirShadowMap;

	protected:
	public:
		// Constructors & Destructor
								SceneLights(const Array32<Light>& lights);
								SceneLights(const SceneLights&) = delete;
		SceneLights&			operator=(const SceneLights&) = delete;
								~SceneLights();

		void					GenerateShadowMaps(DrawBuffer&, GPUBuffer&,
												   FrameTransformBuffer& fTransform,
												   unsigned int drawCount,
												   const RectPrism& viewFrustum);

		void					ChangeLightPos(uint32_t index, IEVector3 position);
		void					ChangeLightType(uint32_t index, LightType);
		void					ChangeLightDir(uint32_t index, IEVector3 direction);
		void					ChangeLightColor(uint32_t index, IEVector3 color);
		void					ChangeLightRadius(uint32_t index, float radius);

		// Access 
};

#endif //__SCENE_H__