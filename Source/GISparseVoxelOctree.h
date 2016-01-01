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

#define GI_DENSE_TEX_COUNT 3
#define GI_DENSE_LEVEL 6
#define GI_DENSE_SIZE 64
#define GI_DENSE_SIZE_CUBE (GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE)

static_assert(GI_DENSE_SIZE >> GI_DENSE_LEVEL == 1, "Pow of Two Mismatch.");
static_assert(GI_DENSE_LEVEL - GI_DENSE_TEX_COUNT > 0, "DENSE_TEX_COUNT_MISMATCH");
static_assert(GI_DENSE_TEX_COUNT >= 1, "Dense Count has to be atleast 1");

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

struct SVOConeParams
{
	// x max traverse distance
	// y tangent(ConeAngle)
	// z tangent(ConeAngle / 2)
	// w sample ratio
	float4 coneParams1;

	// x is intensity factor
	// y sqrt2 (to determine surface lengths)
	// z sqrt3 (worst case diagonal factor)
	// w empty
	float4 coneParams2;
};

struct InvFrameTransform;

class GISparseVoxelOctree
{
	private:
		// Allocator References
		std::vector<GICudaAllocator*>			allocators;			// Page Allocators
		std::vector<const CVoxelGrid*>			allocatorGrids;		// Allocator's Responsible Grids

		CSVOConstants							hSVOConstants;
		CudaVector<CSVOConstants>				dSVOConstants;

		// SVO Data (Sparse)
		StructuredBuffer<CSVONeigIndex>			svoNeigbourBuffer;
		StructuredBuffer<CSVONode>				svoNodeBuffer;
		StructuredBuffer<CSVOMaterial>			svoMaterialBuffer;
		StructuredBuffer<uint32_t>				svoLevelOffsets;
		// SVO Data (Dense)
		GLuint									svoDenseNode;
		GLuint									svoDenseMat;
		GLuint									nodeSampler;
		GLuint									materialSampler;

		// Light Intensity Texture (for SVO GI)
		GLuint									liTexture;

		// Rendering Helpers
		StructuredBuffer<SVOTraceData>			svoTraceData;
		StructuredBuffer<SVOConeParams>			svoConeParams;

		// SVO Ptrs Cuda
		CSVOMaterial*							dSVOMaterial;
		CSVONeigIndex*							dSVONeigbour;
		CSVONode*								dSVOSparse;
		CSVONode*								dSVODense;
		uint32_t*								dSVOOffsets;
		cudaArray_t								dSVODenseNodeArray;
		std::vector<cudaArray_t>				dSVODenseMatArray;
		cudaTextureObject_t						tSVODenseNode;
		cudaSurfaceObject_t						sSVODenseNode;
		std::vector<cudaSurfaceObject_t>		sSVODenseMat;
		
		// Atomic counter and svo level start locations
		CudaVector<uint32_t>					dSVOLevelTotalSizes;
		std::vector<uint32_t>					hSVOLevelTotalSizes;
		CudaVector<uint32_t>					dSVOLevelSizes;
		std::vector<uint32_t>					hSVOLevelSizes;
		
		// Interop Data
		cudaGraphicsResource_t					svoNodeResource;
		cudaGraphicsResource_t					svoNeigbourResource;
		cudaGraphicsResource_t					svoLevelOffsetResource;
		cudaGraphicsResource_t					svoMaterialResource;
		cudaGraphicsResource_t					svoDenseNodeResource;
		cudaGraphicsResource_t					svoDenseTexResource;
		
		// Trace Shaders
		Shader									computeVoxTraceWorld;
		Shader									computeVoxTraceDeferred;
		Shader									computeAO;
		Shader									computeAOSurf;


		void									CreateSurfFromArray(cudaArray_t&,
																	cudaSurfaceObject_t&);
		void									CreateTexFromArray(cudaArray_t&,
																   cudaTextureObject_t&);
		void									CopyFromBufferToTex(cudaArray_t&, unsigned int* dPtr);

		//
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
		double									AmbientOcclusionSurf(DeferredRenderer& dRenderer,
																	 const Camera& camera,
																	 float coneAngle,
																	 float maxDistance,
																	 float sampleDistanceRatio,
																	 float intensityFactor);

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
