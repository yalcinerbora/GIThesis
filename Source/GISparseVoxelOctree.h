#pragma once
/**



*/

#include <cuda.h>
#include <cassert>
#include "COpenGLTypes.h"
#include "CudaVector.cuh"
#include "CSVOTypes.h"
#include "Shader.h"

class SceneI;
class GIVoxelPages;
class GIVoxelCache;
class DeferredRenderer;
class ConeTraceTexture;

#pragma pack(push, 1)
struct OctreeUniforms
{
	IEVector3	worldPos;
	float		baseSpan;

	uint32_t	minSVOLevel;
	uint32_t	denseLevel;
	uint32_t	minCascadeLevel;
	uint32_t	maxSVOLevel;
	
	uint32_t	cascadeCount;
	uint32_t	nodeOffsetDifference;
	uint32_t	gridSize;
	uint32_t	pad0;
};

struct IndirectUniforms
{
	float specularAngleMin;
	float specularAngleMax;
	float diffAngleTanHalf;
	float sampleRatio;
	
	float startOffsetBias;
	float totalDistance;
	float aoIntensity;
	float giIntensity;

	float aoFalloff;
	float pading0;
	float pading1;
	float pading2;
};
#pragma pack(pop)

static constexpr size_t BigSizes[] =
{
	1,					// Root
	8,					// 1 Dense
	64,					// 2 Dense
	512,				// 3 Dense
	4096,				// 4 Dense
	32768,				// 5 Dense
	262144,				// 6 Dense

	256	 * 1024,		// 7
	1	 * 1024 * 1024,	// 8
	4	 * 1024 * 1024,	// 9
	10	 * 1024 * 1024,	// 10
	6	 * 1024 * 1024,	// 11	
	8	 * 1024 * 1024,	// 12
	8	 * 1024 * 1024,	// 13
};


//static constexpr size_t BigSizes[] =
//{
//	1,					// Root
//	8,					// 1 Dense
//	64,					// 2 Dense
//	512,				// 3 Dense
//	4096,				// 4 Dense
//	32768,				// 5 Dense
//	262144,				// 6 Dense
//
//	64 * 1024,	// 7
//	64 * 1024,	// 8
//	100 * 1024,	// 9
//	100 * 1024,	// 10
//	100 * 1024,	// 11	
//	64 * 1024,	// 12
//	64 * 1024,	// 13
//};

class OctreeParameters
{
	public:
		// SVO Const Params
		const uint32_t				DenseLevel;
		const uint32_t				DenseSize;
		const uint32_t				DenseSizeCube;
		const uint32_t				DenseLevelCount;

		const uint32_t				CascadeCount;
		const uint32_t				CascadeBaseLevel;
		const uint32_t				CascadeBaseLevelSize;

		const float					BaseSpan;

		const uint32_t				MinSVOLevel;
		const uint32_t				MaxSVOLevel;

		OctreeParameters(uint32_t denseLevel,
						 uint32_t denseLevelCount,
						 uint32_t cascadeCount,
						 uint32_t cascadeBaseLevel,
						 float baseSpan)
			: DenseLevel(denseLevel)
			, DenseSize(1 << denseLevel)
			, DenseSizeCube(DenseSize * DenseSize * DenseSize)
			, DenseLevelCount(denseLevelCount)
			, CascadeCount(cascadeCount)
			, CascadeBaseLevel(cascadeBaseLevel)
			, CascadeBaseLevelSize(1 << cascadeBaseLevel)
			, BaseSpan(baseSpan)
			, MinSVOLevel(denseLevel - denseLevelCount + 1)
			, MaxSVOLevel(cascadeBaseLevel + cascadeCount - 1)
		{
			assert(static_cast<int>(DenseLevel) - static_cast<int>(DenseLevelCount) > 0);
			assert(DenseLevelCount >= 1);
			assert(CascadeBaseLevel <= 10);
			assert(CascadeCount <= 4);
		}
};

struct LightInjectParameters
{
	const float4 camPos;
	const float3 camDir;

	const float depthNear;
	const float depthFar;
};

class GISparseVoxelOctree
{
	public:
		class ShadowMapsCUDA
		{
			private:
				uint32_t				lightCount;
				size_t					matrixOffset;
				size_t					lightOffset;

				cudaGraphicsResource_t	shadowMapResource;
				cudaGraphicsResource_t	lightBufferResource;

				cudaMipmappedArray_t	shadowMapArray;
				cudaTextureObject_t		tShadowMapArray;

				const CLight*			dLightParamArray;
				const CMatrix4x4* 		dLightVPMatrixArray;
				
			public:
				// Constructors & Destructor
										ShadowMapsCUDA();
										ShadowMapsCUDA(const SceneLights& sLights);
										ShadowMapsCUDA(const ShadowMapsCUDA&) = delete;
										ShadowMapsCUDA(ShadowMapsCUDA&&);
				ShadowMapsCUDA&			operator=(const ShadowMapsCUDA&) = delete;
				ShadowMapsCUDA&			operator=(ShadowMapsCUDA&&);
										~ShadowMapsCUDA();

				void					Map();
				void					Unmap();

				uint32_t				LightCount() const;
				const CLight*			LightParamArray() const;
				const CMatrix4x4*		LightVPMatrices() const;
				cudaTextureObject_t		ShadowMapArray() const;				
		};

	private:
		const OctreeParameters*			octreeParams;
		const SceneI*					scene;

		// Actual SVO Tree
		StructuredBuffer<uint8_t>		oglData;
		size_t							octreeUniformsOffset;
		size_t							indirectUniformsOffset;
		size_t							illumOffsetsOffset;
		size_t							nodeOffset;
		size_t							illumOffset;
		
		// Cuda Image of SVO Tree (Generated from OGL Data)
		// Valid only when
		cudaGraphicsResource_t			gpuResource;
		CudaVector<uint8_t>				cudaData;
		const uint32_t*					dLevelCapacities;
		uint32_t*						dLevelSizes;
		CSVOLevel*						dOctreeLevels;

		// Difference between offsets (since node do not hold dense info except last dense level)
		std::vector<uint32_t>			hLevelSizes;
		std::vector<uint32_t>			hIllumOffsetsAndCapacities;
		std::vector<uint32_t*>			dIllumOffsetPtrs;
		size_t							nodeIllumDifference;

		// Shadowmap interop for light injection
		ShadowMapsCUDA					shadowMaps;

		// Trace Shaders
		Shader							compVoxTraceWorld;
		Shader							compVoxSampleWorld;
		Shader							compGI;

		//
		void							MapOGLData();
		void							UnmapOGLData();
	
		void							PrintSVOLevelUsages(const std::vector<uint32_t>& svoSizes) const;
		double							GenerateHierarchy(bool doTiming,
														  // Page System
														  const GIVoxelPages& pages,
														  // Cache System
														  const GIVoxelCache& caches,
														  // Constants
														  uint32_t batchCount,
														  // Light Injection Related
														  const LightInjectParameters& injectParams,
														  const IEVector3& ambientColor,
														  bool injectOn);
		double							GenNeigbourPointers(bool doTiming);
		double							AverageNodes(bool doTiming);

	protected:

	public:
		// Constructors & Destructor
										GISparseVoxelOctree();
										GISparseVoxelOctree(const OctreeParameters& octreeParams,
															const SceneI* currentScene,
															const size_t sizes[]);
										GISparseVoxelOctree(const GISparseVoxelOctree&) = delete;
										GISparseVoxelOctree(GISparseVoxelOctree&&);
		GISparseVoxelOctree&			operator=(const GISparseVoxelOctree&) = delete;
		GISparseVoxelOctree&			operator=(GISparseVoxelOctree&&);
										~GISparseVoxelOctree();

		// Updates SVO Tree depending on the changes of the allocators
		void							UpdateSVO(// Timing Related
												  double& reconstructTime,
												  double& genPtrTime,
												  double& averageTime,
												  bool doTiming,
												  // Page System
												  const GIVoxelPages& pages,
												  // Cache System
												  const GIVoxelCache& caches,
												  // Constants
												  uint32_t batchCount,
												  const LightInjectParameters& injectParams,
												  const IEVector3& ambientColor,
												  bool injectOn);
		
		void							UpdateOctreeUniforms(const IEVector3& outerCascadePos);
		void							UpdateIndirectUniforms(const IndirectUniforms& indirectUniforms);

		// Traces entire scene with the given ray params
		// Writes results to outputTexture
		// Uses GBuffer to create inital rays (free camera to first bounce)
		double							GlobalIllumination(ConeTraceTexture& coneTex,
														   const DeferredRenderer&,
														   const Camera& camera,
														   const IndirectUniforms&,
														   bool giOn,
														   bool aoOn,
														   bool specularOn);
		// Debug Tracing
		double							DebugTraceSVO(ConeTraceTexture& coneTex,
													  const DeferredRenderer&,
													  const Camera& camera,
													  uint32_t renderLevel,
													  OctreeRenderType);
		double							DebugSampleSVO(ConeTraceTexture& coneTex,
													   const DeferredRenderer&,
													   const Camera& camera,
													   uint32_t renderLevel,
													   OctreeRenderType);

		// Misc
		size_t							MemoryUsage() const;
};