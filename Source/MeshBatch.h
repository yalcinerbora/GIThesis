/**


*/
#ifndef __MESHBATCH_H__
#define __MESHBATCH_H__

#include "MeshBatchI.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"

struct GFGTransform;

struct BatchParams
{
	size_t				materialCount;
	size_t				objectCount;
	size_t				drawCallCount;
	size_t				totalPolygons;
};

class MeshBatch : public MeshBatchI
{
	private:
		
	protected:
		GPUBuffer				batchVertex;
		DrawBuffer				batchDrawParams;
		BatchParams				batchParams;

		std::vector<size_t>		maxVoxelCount;
		float					minSpan;

	public:
		// Constructors & Destructor
								MeshBatch(const char* sceneFileName,
										  float minVoxSpan,
										  const Array32<size_t> maxVoxelCounts,
										  bool isSkeletal);
			
		// Static Files
		static const char*		sponzaFileName;
		static const char*		cornellboxFileName;
		static const char*		sibernikFileName;
		
		static size_t			sponzaVoxelSizes[];
		static size_t			cornellVoxelSizes[];
		static size_t			sibernikVoxelSizes[];

		// Interface
		void					Update(double elapsedS) override;

		DrawBuffer&				getDrawBuffer() override;
		GPUBuffer&				getGPUBuffer() override;

		size_t					VoxelCacheMax(uint32_t level) const override;
		VoxelObjectType			MeshType() const override;

		size_t					ObjectCount() const override;
		size_t					DrawCount() const override;
		size_t					MaterialCount() const override;
		size_t					PolyCount() const override;

		float					MinSpan() const override;

		static void				GenTransformMatrix(IEMatrix4x4& transform,
												   IEMatrix4x4& rotation,
												   const GFGTransform& gfgTransform);
};

#endif //__MESHBATCHSTATIC_H__