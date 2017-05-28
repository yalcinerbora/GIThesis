/**


*/
#ifndef __MESHBATCH_H__
#define __MESHBATCH_H__

#include "MeshBatchI.h"
#include "VertexBuffer.h"
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
		int						repeatCount;

	protected:
		VertexBuffer			batchVertex;
		DrawBuffer				batchDrawParams;
		BatchParams				batchParams;

	public:
		// Constructors & Destructor
								MeshBatch();
								MeshBatch(const std::vector<VertexElement>& vertexDefintion, uint32_t byteStride,
										  const std::vector<std::string>& sceneFiles,
										  uint32_t repeatCount = 1);
								MeshBatch(const MeshBatch&) = delete;
								MeshBatch(MeshBatch&&);
		MeshBatch&				operator=(MeshBatch&&);
		MeshBatch&				operator=(const MeshBatch&) = delete;
								~MeshBatch() = default;
			
		// Interface
		void					Update(double elapsedS) override;

		DrawBuffer&				getDrawBuffer() override;
		VertexBuffer&			getVertexBuffer() override;

		MeshBatchType			MeshType() const override;
        int                     RepeatCount() const override;

		size_t					ObjectCount() const override;
		size_t					DrawCount() const override;
		size_t					MaterialCount() const override;
		size_t					PolyCount() const override;

		static void				GenTransformMatrix(IEMatrix4x4& transform,
												   IEMatrix4x4& rotation,
												   const GFGTransform& gfgTransform);
};

#endif //__MESHBATCHSTATIC_H__