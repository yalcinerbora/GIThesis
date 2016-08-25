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
#include "SceneI.h"

#define GI_DENSE_TEX_COUNT 5
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
	private:
		// Allocator References
		std::vector<GICudaAllocator*>			allocators;			// Page Allocators
		std::vector<const CVoxelGrid*>			allocatorGrids;		// Allocator's Responsible Grids

		CSVOConstants							hSVOConstants;
		CudaVector<CSVOConstants>				dSVOConstants;

		// SVO Data (Sparse)
		StructuredBuffer<CSVONode>				svoNodeBuffer;
		StructuredBuffer<CSVOMaterial>			svoMaterialBuffer;
		StructuredBuffer<uint32_t>				svoLevelOffsets;
		
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
		CLight*									dLightParamArray;
		CMatrix4x4* 							dLightVPArray;
		
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

		static void								CreateSurfFromArray(cudaArray_t&,
																	cudaSurfaceObject_t&);
		static void								CreateTexFromArray(cudaArray_t&,
																   cudaTextureObject_t&);
		static void								CopyFromBufferToTex(cudaArray_t&, 
																	unsigned int* dPtr);
		static void								CreateTexLayeredFromArray(cudaMipmappedArray_t&,
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
												GISparseVoxelOctree();
												GISparseVoxelOctree(const GISparseVoxelOctree&) = delete;
		GISparseVoxelOctree&					operator=(const GISparseVoxelOctree&) = delete;
												~GISparseVoxelOctree();

		// Link Allocators and Adjust Size of the System
		void									LinkAllocators(Array32<GICudaAllocator*> allocators,
															   uint32_t totalCount,
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
#endif //__GICUDASPARSEVOXELOCTREE_H__
