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
#define GI_DENSE_SIZE_CUBE (GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE)

static_assert(GI_DENSE_SIZE >> GI_DENSE_LEVEL == 1, "Pow of Two Mismatch.");

class GICudaAllocator;
class DeferredRenderer;
struct Camera;

enum class SVOTraceType : uint32_t
{
	COLOR,
	OCCULUSION,
	NORMAL
};

struct SVOTree
{
	CSVONode** gLevelNodes;
	CSVOMaterial** gLevelMaterials;
	CSVONode* gLevelNodeCount;

	CSVOMaterial* gDenseMaterial;	
};

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

struct InvFrameTransform;

class GISparseVoxelOctree
{
	private:
		std::vector<GICudaAllocator*>			allocators;			// Page Allocators
		std::vector<const CVoxelGrid*>			allocatorGrids;		// Allocator's Responsible Grids

		CSVOConstants							hSVOConstants;
		CudaVector<CSVOConstants>				dSVOConstants;

		// Texture copied from dSVO dense every frame
		cudaTextureObject_t						tSVODense;
		cudaArray_t								denseArray;

		// SVO Data
		StructuredBuffer<CSVONode>				svoNodeBuffer;
		StructuredBuffer<CSVOMaterial>			svoMaterialBuffer;
		StructuredBuffer<uint32_t>				svoLevelOffsets;

		// Light Intensity Texture (for SVO GI)
		GLuint									liTexture;

		// Rendering Helpers
		StructuredBuffer<SVOTraceData>			svoTraceData;

		// SVO Ptrs
		CSVOMaterial*							dSVOMaterial;
		CSVONode*								dSVODense;
		CSVONode*								dSVOSparse;
		uint32_t*								dSVOOffsets;

		// SVO Mat indices
		uint32_t								matSparseOffset;

		// Atomic counter and svo level start locations
		CudaVector<uint32_t>					dSVOLevelTotalSizes;
		std::vector<uint32_t>					hSVOLevelTotalSizes;
		CudaVector<uint32_t>					dSVOLevelSizes;
		std::vector<uint32_t>					hSVOLevelSizes;
		
		// Interop Data
		cudaGraphicsResource_t					svoNodeResource;
		cudaGraphicsResource_t					svoLevelOffsetResource;
		cudaGraphicsResource_t					svoMaterialResource;
		
		// Trace Shaders
		Shader									computeVoxTraceWorld;
		Shader									computeVoxTraceDeferred;
		Shader									computeAO;

		void									ConstructDense();
		void									ConstructLevel(unsigned int levelIndex,
															   unsigned int allocatorIndex);
		void									ConstructFullAtomic();
		void									ConstructLevelByLevel();
		void									AverageNodes(bool skipLeaf);

		static const GLsizei					TraceWidth;
		static const GLsizei					TraceHeight;

	protected:

	public:
		// Constructors & Destructor
												GISparseVoxelOctree();
												GISparseVoxelOctree(const GISparseVoxelOctree&) = delete;
		GISparseVoxelOctree&					operator=(const GISparseVoxelOctree&) = delete;
												~GISparseVoxelOctree();

		// Link Allocators and Adjust Size of the System
		void									LinkAllocators(Array32<GICudaAllocator*> allocators,
															   uint32_t totalCount,
															   const uint32_t levelCounts[]);

		// Updates SVO Tree depending on the changes of the allocators
		double									UpdateSVO();
		
		// Traces entire scene with the given ray params
		// Writes results to intensity texture
		// Uses GBuffer to create inital rays (free camera to first bounce)
		double									AmbientOcclusion(DeferredRenderer&,
																 const Camera& camera,
																 float coneAngle,
																 float maxDistance,
																 float sampleDistanceRatio);

		double									DebugTraceSVO(DeferredRenderer&,
															  const Camera& camera,
															  uint32_t renderLevel,
															  SVOTraceType);
		double									DebugDeferredSVO(DeferredRenderer& dRenderer,
																 const Camera& camera,
																 uint32_t renderLevel,
																 SVOTraceType type);

		uint64_t								MemoryUsage() const;
		const CSVOConstants&					SVOConsts() const;
};
#endif //__GICUDASPARSEVOXELOCTREE_H__
