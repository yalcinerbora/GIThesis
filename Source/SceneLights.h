/**

*/

#ifndef __SCENELIGHTS_H__
#define __SCENELIGHTS_H__

#include "IEUtility/IEVector4.h"
#include "StructuredBuffer.h"
#include "ArrayStruct.h"
#include <cstdint>

class DrawBuffer;
class GPUBuffer;

struct Light
{
	IEVector4 position;			// position.w is the light type
	IEVector4 direction;		// direction.w holds shadow map index
	IEVector4 color;			// color.a is effecting radius
};

enum class LightType
{
	POINT,
	DIRECTIONAL,
	AREA
};

class SceneLights
{
	private:
		// Sparse texture cubemap array
		// One Shadowmap for each light
		// Directional Lights have one side used (others not allocated)
		// Area Lights only use 5 sides of the cube map
		StructuredBuffer<Light> lightsGPU;
		GLuint					lightShadowMaps;

		// Some Data Related to the scene
		size_t					materialCount;
		size_t					objectCount;
		size_t					drawCallCount;
		size_t					totalPolygons;

	protected:
	public:
		// Constructors & Destructor
								SceneLights(const Array32<Light>& lights);
								SceneLights(const SceneLights&) = delete;
		SceneLights&			operator=(const SceneLights&) = delete;
								~SceneLights();

		void					GenerateShadowMaps(DrawBuffer&, GPUBuffer&);

		void					ChangeLightPos(uint32_t index, IEVector3 position);
		void					ChangeLightType(uint32_t index, LightType);
		void					ChangeLightDir(uint32_t index, IEVector3 direction);
		void					ChangeLightColor(uint32_t index, IEVector3 color);
		void					ChangeLightRadius(uint32_t index, float radius);

		// Access 
};

#endif //__SCENE_H__