#pragma once

#include <cstdint>
#include "StructuredBuffer.h"
#include "CudaVector.cuh"
#include "CVoxelTypes.h"
#include "VoxelDebugVAO.h"

//#pragma pack(push, 1)
//struct ObjGridInfo
//{
//	float span;
//	uint32_t voxCount;
//};
//
//struct VoxelGridInfoGL
//{
//	IEVector4		posSpan;
//	uint32_t		dimension[4];
//};
//#pragma pack(pop)

class Shader;
class MeshBatchI;

class VoxelCacheBatch
{
	public:
		struct Cascade
		{
			CVoxelPos*						dVoxelPos;
			CVoxelNorm*						dVoxelNorm;
			CVoxelAlbedo*					dVoxelAlbedo;
			CVoxelWeight*					dVoxelWeight;
		};

		struct Mesh
		{
			uint32_t						meshVoxelOffset;
			uint32_t						meshVoxelCount;			
		};

		static constexpr char*				CubeGFGFileName = "cube.gfg";
		
	private:
		// Mesh Batch
		MeshBatchI*							batch;
		const std::vector<std::string>*		gfgNames;

		// Actual Data
		CudaVector<uint8_t>					gpuData;

		// Debug Data
		StructuredBuffer<uint8_t>			debugDrawBuffer;
		
		// Per Mesh per Cascade Data
		std::vector<Mesh>					meshData;
		
		// Per Cascade Data
		std::vector<VoxelDebugVAO>			cascadeDebugVAOs;
		std::vector<Cascade>				cascadeDataPointers;
		std::vector<uint32_t>				cascadeVoxelCount;
		std::vector<double>					cascadeVoxelSizes;
		
		bool								isSkeletal;
		bool								glAllocated;
	
		static const std::string			GenVoxelGFGFileName(const std::string& fileName, float span);

		uint32_t							FetchVoxelCount(const std::string& voxelGFGFile);
		size_t								LoadVoxels(size_t offset, const std::string& voxelGFGFile);
		double								CaclulateCascadeMemoryUsageMB(uint32_t cascade) const;

	protected:


	public:
		// Constructors & Destructor
											VoxelCacheBatch();
											VoxelCacheBatch(float minSpan, uint32_t levelCount,
															MeshBatchI& batch,
															const std::vector<std::string>& batchNames);
											VoxelCacheBatch(const VoxelCacheBatch&) = delete;
											VoxelCacheBatch(VoxelCacheBatch&&);
											~VoxelCacheBatch() = default;

		void								AllocateGL(uint32_t cascade,
													   const std::vector<uint32_t>);
		void								DeallocateGL();
		void								Draw(uint32_t cascade,
												 Shader& vDebugVoxel,
												 Shader& fDebugVoxel,
												 const Camera& camera);

		const Cascade&						getCascade(uint32_t cascade) const;
};