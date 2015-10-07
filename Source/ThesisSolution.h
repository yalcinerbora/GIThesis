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
	GI_VOXEL_PAGE,
	GI_VOXEL_CACHE2048,
	GI_VOXEL_CACHE1024,
	GI_VOXEL_CACHE512,
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
		Shader					vertexDebugWorldVoxelCascade;
		Shader					fragmentDebugWorldVoxelCascade;

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
		
		// Utility Buffers
		StructuredBuffer<VoxelGridInfoGL>	gridInfoBuffer;
											
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
																 bool isOuterCascade,
																 uint32_t voxCount);

		 // Voxelizes the scene for a cache level
		double								Voxelize(VoxelObjectCache&,
													 float gridSpan, unsigned int minSpanMultiplier);
		void								LinkCacheWithVoxScene(GICudaVoxelScene&, 
																  VoxelObjectCache&,
																  float coverageRatio);
													 
		// Cuda Segment
		GICudaVoxelScene		voxelScene2048;
		GICudaVoxelScene		voxelScene1024;
		GICudaVoxelScene		voxelScene512;

		static size_t			InitialObjectGridSize;
		static size_t			InitialVoxelBufferSizes;

		// Pre Allocating withput determining total size
		// These are pre calculated
		static size_t			MaxVoxelCacheSize2048;
		static size_t			MaxVoxelCacheSize1024;
		static size_t			MaxVoxelCacheSize512;

	protected:
		
	public:
								ThesisSolution(DeferredRenderer&, const IEVector3& intialCamPos);
								ThesisSolution(const ThesisSolution&) = delete;
		const ThesisSolution&	operator=(const ThesisSolution&) = delete;
								~ThesisSolution();

		// Interface
		bool					IsCurrentScene(SceneI&) override;
		void					Init(SceneI&) override;
		void					Release() override;
		void					Frame(const Camera&) override;
		void					SetFPS(double fpsMS) override;
};
#endif //__THESISSOLUTION_H__