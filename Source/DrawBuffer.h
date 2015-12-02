/**

Draw Buffer

Holds Transformation Matrices,
Holds draw point buffer

*/

#ifndef __DRAWBUFFER_H__
#define __DRAWBUFFER_H__

#include <vector>
#include "GLHeader.h"
#include "Material.h"
#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEMatrix3x3.h"
#include "IEUtility/IEVector4.h"
#include "StructuredBuffer.h"
#include "DrawPoint.h"

#pragma pack(push, 1)
struct ModelTransform
{
	IEMatrix4x4 model;
	IEMatrix4x4 modelRotation;
};

struct AABBData
{
	IEVector4 min;
	IEVector4 max;
};
#pragma pack(pop)

class DrawBuffer
{
	private:
		static uint32_t						initialCapacity;
		
		StructuredBuffer<DrawPointIndexed>	drawPoints;
		StructuredBuffer<ModelTransform>	drawTransforms;
		StructuredBuffer<AABBData>			drawAABBs;
		StructuredBuffer<uint32_t>			modelTransformIndices;

		std::vector<uint32_t>				materialIndex;
		std::vector<Material>				materials;

	protected:
	public:
		// Constructors & Destructor
											DrawBuffer();
											DrawBuffer(const DrawBuffer&) = delete;
		const DrawBuffer&					operator=(const DrawBuffer&) = delete;
											~DrawBuffer() = default;

		// 
		void								AddMaterial(const ColorMaterial&);
		void								AddTransform(const ModelTransform&);
		void								AddDrawCall(const DrawPointIndexed&,
														uint32_t materialIndex,
														uint32_t transformIndex,
														const AABBData& aabb);

		void								SendToGPU();

		StructuredBuffer<ModelTransform>&	getModelTransformBuffer();
		StructuredBuffer<AABBData>&			getAABBBuffer();
		StructuredBuffer<DrawPointIndexed>&	getDrawParamBuffer();
		StructuredBuffer<uint32_t>&			getModelTransformIndexBuffer();

		void								BindMaterialForDraw(uint32_t meshIndex);
		
};
#endif //__DRAWBUFFER_H__