/**


*/
#ifndef __MESHBATCHSTATIC_H__
#define __MESHBATCHSTATIC_H__

#include "MeshBatchI.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"

struct BatchParams
{
	size_t				materialCount;
	size_t				objectCount;
	size_t				drawCallCount;
	size_t				totalPolygons;
};

class MeshBatchStatic : public MeshBatchI
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
								MeshBatchStatic(const char* sceneFileName,
												float minVoxSpan,
												const Array32<size_t> maxVoxelCounts);
			
		// Static Files
		static const char*		sponzaFileName;
		static const char*		cornellboxFileName;
		
		static size_t			sponzaVoxelSizes[];
		static size_t			cornellVoxelSizes[];

		// Interface
		void					Update(double elapsedS) override;

		DrawBuffer&				getDrawBuffer() override;
		GPUBuffer&				getGPUBuffer() override;

		size_t					VoxelCacheMax(uint32_t level) const override;

		size_t					ObjectCount() const override;
		size_t					DrawCount() const override;
		size_t					MaterialCount() const override;
		size_t					PolyCount() const override;

		float					MinSpan() const override;
};

#endif //__MESHBATCHSTATIC_H__