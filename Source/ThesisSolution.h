/**

Solution implementtion

*/

#ifndef __THESISSOLUTION_H__
#define __THESISSOLUTION_H__

#include <vector>
#include <list>
#include <AntTweakBar.h>
#include "SolutionI.h"
#include "Shader.h"
#include "FrameTransformBuffer.h"
#include "GICudaVoxelScene.h"
#include "StructuredBuffer.h"
#include "IEUtility/IEVector3.h"
#include "VoxelRenderTexture.h"
#include "VoxelDebugVAO.h"
#include "DrawBuffer.h"
#include "MeshBatchI.h"
#include "GISparseVoxelOctree.h"

class DeferredRenderer;

#pragma pack(push, 1)
struct ObjGridInfo
{
	float span;
	uint32_t voxCount;
};

struct VoxelGridInfoGL
{
	IEVector4		posSpan;
	uint32_t		dimension[4];
};
#pragma pack(pop)

enum ThesisRenderScheme
{
	GI_DEFERRED,
	GI_LIGHT_INTENSITY,
	GI_SVO_DEFERRED,
	GI_SVO_LEVELS,
	GI_VOXEL_PAGE,
	GI_VOXEL_CACHE2048,
	GI_VOXEL_CACHE1024,
	GI_VOXEL_CACHE512,
	GI_END,
};

class AOBar
{
	public:
	TwBar* bar;
	float angleDegree;
	float sampleFactor;
	float maxDistance;
	float intensity;
	bool hidden;

	AOBar();
	~AOBar();

	void HideBar(bool);
};

struct VoxelObjectCache
{
	StructuredBuffer<ObjGridInfo>			objectGridInfo;
	StructuredBuffer<VoxelNormPos>			voxelNormPos;
	StructuredBuffer<VoxelIds>				voxelIds;
	StructuredBuffer<VoxelRenderData>		voxelRenderData;
	StructuredBuffer<uint32_t>				voxelCacheUsageSize;
	VoxelDebugVAO							voxelVAO;
	
	uint32_t								batchVoxCacheCount;
	double									batchVoxCacheSize;

	VoxelObjectCache(size_t objectCount, size_t voxelCount)
		: objectGridInfo(objectCount)
		, voxelNormPos(voxelCount)
		, voxelIds(voxelCount)
		, voxelRenderData(voxelCount)
		, voxelCacheUsageSize(1)
		, voxelVAO(voxelNormPos, voxelIds, voxelRenderData)
	{
		voxelCacheUsageSize.AddData(0);
	}

	VoxelObjectCache(VoxelObjectCache&& other)
		: objectGridInfo(std::move(other.objectGridInfo))
		, voxelNormPos(std::move(other.voxelNormPos))
		, voxelIds(std::move(other.voxelIds))
		, voxelRenderData(std::move(other.voxelRenderData))
		, voxelCacheUsageSize(std::move(other.voxelCacheUsageSize))
		, voxelVAO(std::move(other.voxelVAO))
		, batchVoxCacheCount(other.batchVoxCacheCount)
		, batchVoxCacheSize(other.batchVoxCacheSize)
	{}

	VoxelObjectCache(const VoxelObjectCache&) = delete;
};

struct SceneVoxCache
{
	uint32_t						depth;
	std::vector<VoxelObjectCache>	cache;

	uint32_t						voxOctreeCount;
	double							voxOctreeSize;

	uint32_t						totalCacheCount;
	double							totalCacheSize;

	SceneVoxCache() = default;

	SceneVoxCache(SceneVoxCache&& other)
	: depth(other.depth)
	, cache(std::move(other.cache))
	, voxOctreeCount(other.voxOctreeCount)
	, voxOctreeSize(other.voxOctreeSize)
	, totalCacheCount(other.totalCacheCount)
	, totalCacheSize(other.totalCacheSize)
	{}

	SceneVoxCache(const SceneVoxCache&) = delete;
};

class ThesisSolution : public SolutionI
{
	private:
		SceneI*					currentScene;

		DeferredRenderer&		dRenderer;

		// Voxel Render Shaders
		Shader					vertexDebugVoxel;
		Shader					vertexDebugWorldVoxel;
		Shader					fragmentDebugVoxel;

		// Voxelization Shaders
		Shader					vertexVoxelizeObject;
		Shader					geomVoxelizeObject;
		Shader					fragmentVoxelizeObject;
		Shader					computeVoxelizeCount;
		Shader					computePackObjectVoxels;
		Shader					computeDetermineVoxSpan;

		FrameTransformBuffer	cameraTransform;

		// Voxel Cache for each cascade
		std::vector<SceneVoxCache>			voxelCaches;

		// Cuda Stuff
		std::vector<GICudaVoxelScene>		voxelScenes;
		GISparseVoxelOctree					voxelOctree;

		// Utility(Debug) Buffers
		StructuredBuffer<VoxelGridInfoGL>	gridInfoBuffer;
		StructuredBuffer<VoxelNormPos>		voxelNormPosBuffer;
		StructuredBuffer<uchar4>			voxelColorBuffer;

		cudaGraphicsResource_t				vaoNormPosResource;
		cudaGraphicsResource_t				vaoRenderResource;

		// GUI								
		TwBar*								bar;
		bool								giOn;
		double								frameTime;
		double								ioTime;
		double								transformTime;
		double								svoTime;
		double								debugVoxTransferTime;
											
		ThesisRenderScheme					renderScheme;
		static const TwEnumVal				renderSchemeVals[];
		TwType								renderType;
											
		// Debug Rendering					
		void								DebugRenderVoxelCache(const Camera& camera, 
																  SceneVoxCache&);
		void								DebugRenderVoxelPage(const Camera& camera,
																 VoxelDebugVAO& pageVoxels,
																 const CVoxelGrid& voxGrid,
																 uint32_t offset,
																 uint32_t voxCount);
		double								DebugRenderSVO(const Camera& camera);

		 // Voxelizes the scene for a cache level
		double								Voxelize(VoxelObjectCache&,
													 MeshBatchI* batch,
													 float gridSpan, 
													 unsigned int minSpanMultiplier,
													 bool isInnerCascade);
		void								LinkCacheWithVoxScene(GICudaVoxelScene&, 
																  SceneVoxCache&,
																  float coverageRatio);
													 
		// Cuda Segment
		static const size_t		InitialObjectGridSize;
		static const size_t		InitialVoxelBufferSizes;

		// Pre Allocating withput determining total size
		// These are pre calculated
		static const size_t		MaxVoxelCacheSize2048;
		static const size_t		MaxVoxelCacheSize1024;
		static const size_t		MaxVoxelCacheSize512;

		AOBar					aoBar;
		void					LevelIncrement();
		void					LevelDecrement();
		void					TraceTypeInc();
		void					TraceTypeDec();
		unsigned int			svoRenderLevel;
		unsigned int			traceType;

	protected:
		
	public:
								ThesisSolution(DeferredRenderer&, const IEVector3& intialCamPos);
								ThesisSolution(const ThesisSolution&) = delete;
		const ThesisSolution&	operator=(const ThesisSolution&) = delete;
								~ThesisSolution();

		// Globals
		static const float		CascadeSpan;
		static const uint32_t	CascadeDim;

		// Interface
		bool					IsCurrentScene(SceneI&) override;
		void					Init(SceneI&) override;
		void					Release() override;
		void					Frame(const Camera&) override;
		void					SetFPS(double fpsMS) override;

		static void				LevelIncrement(void*);
		static void				LevelDecrement(void*);

		static void				TraceIncrement(void*);
		static void				TraceDecrement(void*);
};
#endif //__THESISSOLUTION_H__