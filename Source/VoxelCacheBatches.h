#pragma once

#include <cstdint>
#include "StructuredBuffer.h"
#include "CudaVector.cuh"
#include "CVoxelTypes.h"
#include "VoxelDebugVAO.h"

struct Camera;
class Shader;
class MeshBatchI;

class VoxelCacheBatches
{
	public:
		struct CascadeBatch
		{
			CMeshVoxelInfo*					dMeshVoxelInfo;
			CVoxelPos*						dVoxelPos;
			CVoxelNorm*						dVoxelNorm;
			CVoxelAlbedo*					dVoxelAlbedo;
			CVoxelWeights*					dVoxelWeight;
		};

		struct Mesh
		{
			uint32_t						meshVoxelOffset;
			uint32_t						meshVoxelCount;			
		};

		static constexpr char*				CubeGFGFileName = "cube.gfg";
		
	private:
		// Mesh Batch
		const std::vector<MeshBatchI*>*		batches;

		// Actual Data
		CudaVector<uint8_t>					gpuData;

		// Debug Data
		StructuredBuffer<uint8_t>			debugDrawBuffer;
		
		// Per Mesh per Cascade Data
		std::vector<Mesh>					meshData;
		
		// Per Cascade Data
//		std::vector<VoxelDebugVAO>			cascadeDebugVAOs;
		std::vector<CascadeBatch>			cascadeDataPointers;
		std::vector<uint32_t>				cascadeVoxelCount;
		std::vector<double>					cascadeVoxelSizes;
		
		bool								glAllocated;
	
		static const std::string			GenVoxelGFGFileName(const std::string& fileName, float span);

		size_t								FetchFileVoxelSize(const std::string& voxelGFGFile, bool isSkeletal, int repeatCount = 1);
		size_t								LoadBatchVoxels(size_t gpuBufferOffset, float currentSpan,
															const MeshBatchI* batch,
															const std::vector<std::string>& gfgFiles);
		static double						CaclulateCascadeMemoryUsageMB(uint32_t voxelCount, uint32_t meshCount, bool isSkeletal);

	protected:


	public:
		// Constructors & Destructor
											VoxelCacheBatches();
											VoxelCacheBatches(float minSpan, uint32_t levelCount,
															  const std::vector<MeshBatchI*>* batches,
															  const std::vector<std::vector<std::string>>& batchFileNames);
											VoxelCacheBatches(const VoxelCacheBatches&) = delete;
											VoxelCacheBatches(VoxelCacheBatches&&);
											~VoxelCacheBatches() = default;

		void								AllocateGL(uint32_t cascade);
		void								DeallocateGL();
		void								Draw(uint32_t cascade,
												 Shader& vDebugVoxel,
												 Shader& fDebugVoxel,
												 const Camera& camera);

		const CascadeBatch&					getCascade(uint32_t cascade, uint32_t batch) const;
};