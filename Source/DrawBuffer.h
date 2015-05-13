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

struct DrawPointIndexed;

#pragma pack(push, 1)
struct ModelTransform
{
	IEMatrix4x4 model;
	// Because of the std140 rule (each column of 3x3 matrix should be 
	// interleaved vec4 boundaries
	IEVector4 modelRotationC1;
	IEVector4 modelRotationC2;
	IEVector4 modelRotationC3;

	// TODO: This is a bullshit solution it only works on my cards but w/e
	// OffsetAlignment is 256 on my GTX660Ti and Quadro 4000
	uint8_t offset[256 - sizeof(IEMatrix4x4) - (sizeof(IEVector4) * 3)];
};

struct AABBData
{
	IEVector4 min;
	IEVector4 max;

	// TODO: This is a bullshit solution it only works on my cards but w/e
	// OffsetAlignment is 256 on my GTX660Ti and Quadro 4000
	//uint8_t offset[256 - sizeof(IEVector4) * 2];
};
#pragma pack(pop)

class DrawBuffer
{
	private:
		static uint32_t						initialCapacity;
		
		StructuredBuffer<DrawPointIndexed>	drawPoints;
		StructuredBuffer<ModelTransform>	drawTransforms;
		StructuredBuffer<AABBData>			drawAABBs;

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
		void								AddDrawCall(const DrawPointIndexed&,
														uint32_t materialIndex,
														const ModelTransform& modelTransform,
														const AABBData& aabb);

		void								SendToGPU();

		StructuredBuffer<ModelTransform>&	getModelTransformBuffer();
		StructuredBuffer<AABBData>&			getAABBBuffer();
		StructuredBuffer<DrawPointIndexed>&	getDrawParamBuffer();

		void								BindMaterialForDraw(uint32_t meshIndex);
		
};
#endif //__DRAWBUFFER_H__