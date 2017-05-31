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
#include "Globals.h"

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
											DrawBuffer(DrawBuffer&&);
		DrawBuffer&							operator=(DrawBuffer&&);
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
		void									LockAndLoad();
		void									SendModelTransformToGPU(uint32_t offset = 0, uint32_t size = std::numeric_limits<uint32_t>::max());
		ModelTransform&							getModelTransform(uint32_t transformId);

		GLuint									getGLBuffer();

		size_t									getModelTransformOffset() const;
		size_t									getAABBOffset() const;
		size_t									getDrawParamOffset() const;		
		size_t									getModelTransformIndexOffset() const;

		size_t									getDrawPointCount() const;
		size_t									getModelTransformCount() const;
		size_t									getAABBCount() const;
		size_t									getModelTransformIndexCount() const;
		size_t									getMaterialCount() const;

		const AABBData&							getAABB(uint32_t drawId) const;
		//const std::vector<DrawPointIndexed>&	getCPUDrawPoints() const;
		//const std::vector<ModelTransform>&	getCPUModelTransforms() const;
		//const std::vector<AABBData>&			getCPUAABBs() const;
		//const std::vector<uint32_t>&			getCPUModelTransformIndices() const;
		//const std::vector<Material>&			getMaterials() const;

		void									BindAsDrawIndirectBuffer();
		void									BindAABB(GLuint bindPoint);
		void									BindModelTransform(GLuint bindPoint);
		void									BindModelTransformIndex(GLuint bindPoint);

		void									DrawCallSingle(GLuint drawId);
		void									DrawCallMulti();
		void									DrawCallMultiState();

		void									BindMaterialForDraw(uint32_t drawId);
		void									RepeatDrawCalls(uint32_t instanceCount);
};
#endif //__DRAWBUFFER_H__