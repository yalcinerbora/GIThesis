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

enum class VoxelObjectType : uint32_t
{
	STATIC,			// Object does not move
	DYNAMIC,		// Object does move (with transform matrices)
	SKEL_DYNAMIC,	// Object moves with weighted transformation matrices
	MORPH_DYNAMIC,	// Object moves with morph targets (each voxel has their adjacent vertex morphs weighted)
};

struct VoxelInfo
{
	uint32_t	sceneVoxCacheCount;
	double		sceneVoxCacheSize;

	uint32_t	sceneVoxOctreeCount;
	double		sceneVoxOctreeSize;

};

enum ThesisRenderScheme
{
	GI_DEFERRED,
	GI_LIGHT_INTENSITY,
	GI_SVO_LEVELS,
	GI_VOXEL_PAGE,
	GI_VOXEL_CACHE2048,
	GI_VOXEL_CACHE1024,
	GI_VOXEL_CACHE512,
	GI_END,
};

struct VoxelObjectCache
{
	StructuredBuffer<ObjGridInfo>			objectGridInfo;
	StructuredBuffer<VoxelNormPos>			voxelNormPos;
	StructuredBuffer<VoxelIds>				voxelIds;
	StructuredBuffer<VoxelRenderData>		voxelRenderData;
	StructuredBuffer<uint32_t>				voxelCacheUsageSize;
	VoxelDebugVAO							voxelVAO;
	VoxelInfo								voxInfo;

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
		VoxelObjectCache		cache2048;
		VoxelObjectCache		cache1024;
		VoxelObjectCache		cache512;
		
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
		void								DebugRenderVoxelCache(const Camera& camera, VoxelObjectCache&);
		void								DebugRenderVoxelPage(const Camera& camera,
																 VoxelDebugVAO& pageVoxels,
																 const CVoxelGrid& voxGrid,
																 uint32_t offset,
																 uint32_t voxCount);
		double								DebugRenderSVO(const Camera& camera);

		 // Voxelizes the scene for a cache level
		double								Voxelize(VoxelObjectCache&,
													 float gridSpan, 
													 unsigned int minSpanMultiplier,
													 bool isInnerCascade);
		void								LinkCacheWithVoxScene(GICudaVoxelScene&, 
																  VoxelObjectCache&,
																  float coverageRatio);
													 
		// Cuda Segment
		GICudaVoxelScene		voxelScene2048;
		GICudaVoxelScene		voxelScene1024;
		GICudaVoxelScene		voxelScene512;
		GISparseVoxelOctree		voxelOctree;

		static const size_t		InitialObjectGridSize;
		static const size_t		InitialVoxelBufferSizes;

		// Pre Allocating withput determining total size
		// These are pre calculated
		static const size_t		MaxVoxelCacheSize2048;
		static const size_t		MaxVoxelCacheSize1024;
		static const size_t		MaxVoxelCacheSize512;

		void					LevelIncrement();
		void					LevelDecrement();
		unsigned int			svoRenderLevel;

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
};
#endif //__THESISSOLUTION_H__