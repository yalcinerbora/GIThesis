#pragma once
/**



*/

#include <cuda.h>
#include "CudaVector.cuh"
#include "SceneLights.h"

#include "CSVOTypes.h"
#include "Shader.h"
#include "SceneI.h"

class GICudaAllocator;
class DeferredRenderer;
struct Camera;

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

struct InjectParams
{
	bool inject;
	float span;
	float3 outerCascadePos;

	float4 camPos;
	float3 camDir;
	
	float depthNear;
	float depthFar;


	
	unsigned int lightCount;
};

struct InvFrameTransform;

class GISparseVoxelOctree
{
	public:
		const OctreeParameters&					octreeParams;

	private:

		CSVOConstants							hSVOConstants;
		CudaVector<CSVOConstants>				dSVOConstants;

		//// SVO Data (Sparse)
		//StructuredBuffer<CSVONode>				svoNodeBuffer;
		//StructuredBuffer<CSVOMaterial>			svoMaterialBuffer;
		//StructuredBuffer<uint32_t>				svoLevelOffsets;
		
        // SVO Data (Dense)
		GLuint									svoDenseNode;
		GLuint									svoDenseMat;
		GLuint									nodeSampler;
		GLuint									materialSampler;
		GLuint									gaussSampler;

		// Light Intensity Texture (for SVO GI)
		GLuint									traceTexture;
		GLuint									gaussTex;
		GLuint									edgeTex;

		// Rendering Helpers
		StructuredBuffer<SVOTraceData>			svoTraceData;
		StructuredBuffer<SVOConeParams>			svoConeParams;

		// SVO Ptrs Cuda
		CSVOMaterial*							dSVOMaterial;
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
		cudaGraphicsResource_t					svoLevelOffsetResource;
		cudaGraphicsResource_t					svoMaterialResource;
		cudaGraphicsResource_t					svoDenseNodeResource;
		cudaGraphicsResource_t					svoDenseTexResource;
		
		cudaGraphicsResource_t					sceneShadowMapResource;
		cudaGraphicsResource_t					sceneLightParamResource;
		cudaGraphicsResource_t					sceneVPMatrixResource;

		// Shadows
		cudaMipmappedArray_t					shadowMapArray;
		cudaTextureObject_t						tShadowMapArray;
		//CLight*								dLightParamArray;
		//CMatrix4x4* 							dLightVPArray;
		
		// Trace Shaders
		Shader									computeVoxTraceWorld;
		Shader									computeVoxTraceDeferred;
		Shader									computeVoxTraceDeferredLerp;
		Shader									computeAO;
		Shader									computeGI;
		Shader									computeGauss32;
		Shader									computeEdge;
		Shader									computeAOSurf;
		Shader									computeLIApply;

		void									CreateSurfFromArray(cudaArray_t&,
																	cudaSurfaceObject_t&);
		void									CreateTexFromArray(cudaArray_t&,
																   cudaTextureObject_t&);
		void									CopyFromBufferToTex(cudaArray_t&,
																	unsigned int* dPtr);
		void									CreateTexLayeredFromArray(cudaMipmappedArray_t&,
																		  cudaTextureObject_t&);

		//
		void									ConstructDense();
		void									ConstructLevel(unsigned int levelIndex,
															   unsigned int allocatorIndex);
		double									ConstructFullAtomic(const IEVector3& ambientColor, const InjectParams& p);
		double									ConstructLevelByLevel(const IEVector3& ambientColor, const InjectParams& p);
		double									LightInject(InjectParams,
															const std::vector<IEMatrix4x4>& projMatrices,
															const std::vector<IEMatrix4x4>& invViewProj);
		double									AverageNodes();

		static const GLsizei					TraceWidth;
		static const GLsizei					TraceHeight;

	protected:

	public:
		// Constructors & Destructor
												GISparseVoxelOctree(const OctreeParameters& octreeParams);
												GISparseVoxelOctree(const GISparseVoxelOctree&) = delete;
		GISparseVoxelOctree&					operator=(const GISparseVoxelOctree&) = delete;
												~GISparseVoxelOctree();

		// Link Allocators and Adjust Size of the System
		void									LinkAllocators(std::vector<GICudaAllocator*> allocators,
															   const uint32_t levelCounts[]);

		// Link
		void									LinkSceneShadowMaps(SceneI* scene);

		// Updates SVO Tree depending on the changes of the allocators
		void									UpdateSVO(double& reconstTime,
														  double& injectTime,
														  double& averageTime,
														  const IEVector3& ambientColor,
														  const InjectParams&,
														  const std::vector<IEMatrix4x4>& lightProjMatrices,
														  const std::vector<IEMatrix4x4>& lightInvViewProjMatrices);
		
		// Traces entire scene with the given ray params
		// Writes results to intensity texture
		// Uses GBuffer to create inital rays (free camera to first bounce)
		double									AmbientOcclusion(DeferredRenderer& dRenderer,
																 const Camera& camera,
																 float coneAngle,
																 float maxDistance,
																 float falloffFactor,
																 float sampleDistanceRatio,
																 float intensityFactor);
		double									GlobalIllumination(DeferredRenderer& dRenderer,
																   const Camera& camera,
																   SceneI& scene,
																   float coneAngle,
																   float maxDistance,
																   float falloffFactor,
																   float sampleDistanceRatio,
																   float intensityFactorAO,
																   float intensityFactorGI,
																   bool giOn,
																   bool aoOn,
																   bool specular);

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
		uint32_t								MinLevel() const;
		uint32_t								MaxLevel() const;
};
