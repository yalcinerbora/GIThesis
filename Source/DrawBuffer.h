/**

Draw Buffer

Holds Transformation Matrices,
Holds draw point buffer

*/

#ifndef __DRAWBUFFER_H__
#define __DRAWBUFFER_H__

#include <map>
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
		size_t							drawPointOffset;
		size_t							modelTransformOffset;
		size_t							aabbOffset;
		size_t							modelTransformIndexOffset;

		bool							locked;

		// CPU Image of GPU Data
		std::vector<DrawPointIndexed>	cpuDrawPoints;
		std::vector<ModelTransform>		cpuModelTransforms;		
		std::vector<AABBData>			cpuAABBs;
		std::vector<uint32_t>			cpuModelTransformIndices;

		// GPU Data (packed)
		StructuredBuffer<uint8_t>		gpuData;

		// Material Related
		std::vector<uint32_t>			drawMaterialIndex;
		std::vector<Material>			materials;

	protected:
	public:
		// Constructors & Destructor
											DrawBuffer();
											DrawBuffer(const DrawBuffer&) = delete;
		DrawBuffer&							operator=(const DrawBuffer&) = delete;
											~DrawBuffer() = default;

		// 
		uint32_t							AddMaterial(const ColorMaterial&);
		uint32_t							AddTransform(const ModelTransform&);
		uint32_t							AddDrawCall(const DrawPointIndexed&,
														uint32_t materialIndex,
														uint32_t transformIndex,
														const AABBData& aabb);

		// Locks Draw Call Addition and Loads data to GPU
		void								LockAndLoad();
		void								SendModelTransformToGPU(uint32_t offset = 0, uint32_t size = std::numeric_limits<uint32_t>::max());
		ModelTransform&						ModelTransformBuffer(uint32_t transformId);

		//size_t							getModelTransformOffset() const;
		//size_t							getAABBOffset() const;
		//size_t							getDrawParamOffset() const;
		//size_t							getModelTransformIndexOffset() const;

		void								BindAsDrawIndirectBuffer();
		void								BindAABB(GLuint bindPoint);
		void								BindModelTransform(GLuint bindPoint);
		
		void								DrawCallSingle(GLuint drawId);
		void								DrawCallMulti();
		void								DrawCallMultiState();

		void								BindMaterialForDraw(uint32_t drawId);		
};
#endif //__DRAWBUFFER_H__