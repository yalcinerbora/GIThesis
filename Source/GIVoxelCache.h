#pragma once

#include <cstdint>
#include "StructuredBuffer.h"
#include "CudaVector.cuh"
#include "CVoxelTypes.h"
#include "VoxelVAO.h"
#include "Shader.h"
#include "Globals.h"

struct Camera;
class Shader;
class MeshBatchI;

class GIVoxelCache
{
	public:
		static constexpr char*					CubeGFGFileName = "cube.gfg";
		
	private:
		// Mesh Batch
		const std::vector<MeshBatchI*>*			batches;
		float									baseSpan;

		// Actual Data
		CudaVector<uint8_t>						gpuData;
		CudaVector<BatchVoxelCache>				dDeviceCascadePtrs;
		std::vector<BatchVoxelCache>			hDeviceCascadePtrs;

		// Shaders
		Shader									vRenderVoxelSkel;
		Shader									vRenderVoxel;
		Shader									fRenderVoxel;

		// Loaded only when Drawing (Debug)
		StructuredBuffer<uint8_t>				debugDrawBuffer;
		std::vector<std::vector<MeshVoxelInfo>>	meshData;
		std::vector<VoxelVAO>					debugVAO;
		int32_t									currentCascade;
		uint32_t								cubeIndexCount;

		// Size Storage
		std::vector<uint32_t>					cascadeVoxelCount;
		std::vector<size_t>						cascadeVoxelSizes;
			
		static const std::string				GenVoxelGFGFileName(const std::string& fileName, float span);

		size_t									FetchFileVoxelSize(const std::string& voxelGFGFile, const MeshBatchI* batch);
		size_t									LoadBatchVoxels(size_t gpuBufferOffset, float currentSpan,
																const MeshBatchI* batch,
																const std::vector<std::string>& gfgFiles);
		static size_t							CaclulateCascadeMemoryUsage(uint32_t voxelCount, 
																			uint32_t meshCount, 
																			bool isSkeletal);

	protected:
	public:
		// Constructors & Destructor
												GIVoxelCache();
												GIVoxelCache(float baseSpan, uint32_t levelCount,
															 const std::vector<MeshBatchI*>* batches,
															 const std::vector<std::vector<std::string>>& batchFileNames);
												GIVoxelCache(const GIVoxelCache&) = delete;
												GIVoxelCache(GIVoxelCache&&);
		GIVoxelCache&							operator=(GIVoxelCache&&);
		GIVoxelCache&							operator=(const GIVoxelCache&) = delete;
												~GIVoxelCache() = default;
												
		// Debug Related
		void									AllocateGL(uint32_t cascade);
		void									DeallocateGL();
		double									Draw(bool doTiming,
													 const Camera& camera,
													 VoxelRenderType renderType);

		// Utility	
		const std::vector<CMeshVoxelInfo>		CopyMeshObjectInfo(uint32_t cascadeId, uint32_t batchId) const;
		const std::vector<BatchVoxelCache>&		getDeviceCascadePointersHost() const;
		const CudaVector<BatchVoxelCache>&		getDeviceCascadePointersDevice() const;
};