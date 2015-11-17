/**



*/

#ifndef __GICUDASPARSEVOXELOCTREE_H__
#define __GICUDASPARSEVOXELOCTREE_H__

#include <cuda.h>
#include "CudaVector.cuh"
#include "SceneLights.h"
#include "VoxelDebugVAO.h"
#include "GICudaVoxelScene.h"
#include "CSVOTypes.cuh"
#include "Shader.h"

#define GI_DENSE_LEVEL 6
#define GI_DENSE_SIZE 64

class GICudaAllocator;
struct Camera;

struct SVOTraceData
{
	// xyz gridWorldPosition
	// w is gridSpan
	float4 worldPosSpan;

	// x is grid dimension
	// y is grid depth
	// z is dense dimension
	// w is dense depth
	uint4 dimDepth;

	// x is cascade count
	// y is node sparse offet
	// z is material sparse offset
	uint4 offsetCascade;
};

struct CameraTraceData
{
	float4 camPos;
	float4 camDir;
	float4 camUp;
};

class GISparseVoxelOctree
{
	private:
		std::vector<GICudaAllocator*>			allocators;			// Page Allocators
		std::vector<CVoxelGrid>					allocatorGrids;		// Allocator's Responsible Grids

		CSVOConstants							hSVOConstants;
		CudaVector<CSVOConstants>				dSVOConstants;

		// Debug Stuff
		StructuredBuffer<VoxelNormPos>			vaoNormPosData;
		StructuredBuffer<uchar4>				vaoColorData;

		// Texture copied from dSVO dense every frame
		cudaTextureObject_t						tSVODense;
		cudaArray_t								denseArray;

		// SVO Data
		StructuredBuffer<CSVONode>				svoNodeBuffer;
		StructuredBuffer<CSVOMaterial>			svoMaterialBuffer;

		// Rendering Helpers
		StructuredBuffer<SVOTraceData>			svoTraceData;
		StructuredBuffer<CameraTraceData>		camTraceData;

		// SVO Ptrs
		CSVOMaterial*							dSVOMaterial;
		CSVONode*								dSVODense;
		CSVONode*								dSVOSparse;

		// SVO Mat indices
		uint32_t								matSparseOffset;

		// Atomic counter and svo level start locations
		CudaVector<unsigned int>				dSVONodeAllocator;
		CudaVector<unsigned int>				dSVOLevelSizes;
		std::vector<unsigned int>				hSVOLevelSizes;
		std::vector<unsigned int>				hSVOLevelOffsets;

		// Interop Data
		cudaGraphicsResource_t					svoNodeResource;
		cudaGraphicsResource_t					svoMaterialResource;
		
		// Trace Shaders
		Shader									computeVoxTraceWorld;

		void									ConstructDense();
		void									ConstructLevel(unsigned int levelIndex,
															   unsigned int allocatorIndex);
		void									ConstructFullAtomic();
		void									ConstructLevelByLevel();
		void									AverageNodes(bool orderedNodes);

	protected:

	public:
		// Constructors & Destructor
												GISparseVoxelOctree();
												GISparseVoxelOctree(const GISparseVoxelOctree&) = delete;
		GISparseVoxelOctree&					operator=(const GISparseVoxelOctree&) = delete;
												~GISparseVoxelOctree();

		// Link Allocators and Adjust Size of the System
		void									LinkAllocators(GICudaAllocator** newAllocators,
															   size_t allocatorSize,
															   float sceneMultiplier);

		// Updates SVO Tree depending on the changes of the allocators
		double									UpdateSVO();
		
		// Traces entire scene with the given ray params
		// Writes results to intensity texture
		// Uses GBuffer to create inital rays (free first bounce)
		double									ConeTrace(GLuint depthBuffer,
														  GLuint normalBuffer,
														  GLuint colorBuffer,
														  const Camera& camera);

		double									DebugTraceSVO(GLuint writeImage,
															  const Camera& cam,
															  const uint2 & imgDim);

		uint64_t								MemoryUsage() const;
};
#endif //__GICUDASPARSEVOXELOCTREE_H__
