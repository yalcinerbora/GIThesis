#include "IEUtility/IETimer.h"
#include "VoxelCacheBatches.h"
#include "GFG/GFGFileLoader.h"
#include "MeshBatchI.h"
#include "Macros.h"
#include <sstream>

const std::string VoxelCacheBatches::GenVoxelGFGFileName(const std::string& fileName, float span)
{
	// Voxel File Name is "meshFileName"_vox_0.6.gfg
	size_t startPos = fileName.find_last_of("\\/");
	startPos = (startPos == std::string::npos) ? 0 : startPos;
	size_t endPos = fileName.find_last_of(".");

	std::string fileNameOnly = fileName.substr(startPos + 1, endPos);
	std::ostringstream voxPrefix;
	voxPrefix << fileNameOnly << "_vox_" << span << ".gfg";	
	return voxPrefix.str();
}

size_t VoxelCacheBatches::FetchFileVoxelSize(const std::string& voxelGFGFile, 
											 bool isSkeletal, int repeatCount)
{
	std::ifstream stream(voxelGFGFile, std::ios_base::in | std::ios_base::binary);
	GFGFileReaderSTL stlFileReader(stream);
	GFGFileLoader gfgFile(&stlFileReader);

	// There are total of Two Meshes
	// Last gfg "mesh" holds object information
	// Other one holds voxels
	GFGFileError e = gfgFile.ValidateAndOpen();
	const auto& header = gfgFile.Header();
	assert(e == GFGFileError::OK);
	assert(gfgFile.Header().meshes.size() == 2);

	const auto& meshObjCount = header.meshes.back();
	uint32_t objCount = static_cast<uint32_t>(meshObjCount.headerCore.vertexCount);
	std::vector<uint8_t> objectInfoData(gfgFile.MeshVertexDataSize(1));
	gfgFile.MeshVertexData(objectInfoData.data(), 1);

	MeshVoxelInfo* vInfo = reinterpret_cast<MeshVoxelInfo*>(objectInfoData.data());
	size_t voxelCount = vInfo[objCount - 1].voxOffset + vInfo[objCount - 1].voxCount;

	size_t singleSize = objCount * sizeof(MeshVoxelInfo) +
						voxelCount * (sizeof(CVoxelPos) +
									  sizeof(CVoxelNorm) +
									  sizeof(CVoxelAlbedo) +
									  (isSkeletal) ? sizeof(CVoxelWeights) : 0);
	return singleSize * repeatCount;
}

size_t VoxelCacheBatches::LoadBatchVoxels(size_t gpuBufferOffset, float currentSpan,
										  const MeshBatchI* batch,
										  const std::vector<std::string>& gfgFiles)
{
	// Batch Data
	std::vector<MeshVoxelInfo> voxelInfo;
	std::vector<CVoxelPos> voxelPos;
	std::vector<CVoxelNorm> voxelNorm;
	std::vector<CVoxelAlbedo> voxelAlbedo;
	std::vector<CVoxelWeights> voxelWeights;

	// Load each GFG one by one to temp vectors then push to gpu
	uint32_t voxelBatchOffset = 0;
	uint32_t totalVoxelCount = 0;
	for(const std::string& fileName : gfgFiles)	
	{
		const auto voxelFile = GenVoxelGFGFileName(fileName, currentSpan);
		
		std::ifstream stream(voxelFile, std::ios_base::in | std::ios_base::binary);
		GFGFileReaderSTL stlFileReader(stream);
		GFGFileLoader gfgFile(&stlFileReader);

		// There are total of Two Meshes
		// Last gfg "mesh" holds object information
		// Other one holds voxels
		GFGFileError e = gfgFile.ValidateAndOpen();
		const auto& header = gfgFile.Header();
		assert(e == GFGFileError::OK);
		assert(gfgFile.Header().meshes.size() == 2);

		// Repeat Load
		for(int i = 0; i < batch->RepeatCount(); i++)
		{
			const auto& meshObjCount = header.meshes.back();
			std::vector<MeshVoxelInfo> objectInfoData(gfgFile.MeshVertexDataSize(1));
			gfgFile.MeshVertexData(reinterpret_cast<uint8_t*>(objectInfoData.data()), 1);

			// Some Validation
			const auto& component = meshObjCount.components[i];
			assert(component.dataType == GFGDataType::UINT32_2);
			assert(sizeof(MeshVoxelInfo) == GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)]);
			assert(component.internalOffset == 0);
			assert(component.logic == GFGVertexComponentLogic::POSITION);
			assert(component.stride == sizeof(MeshVoxelInfo));
			assert(static_cast<uint32_t>(meshObjCount.headerCore.vertexCount) == batch->DrawCount());

			uint32_t voxelCount = 0;
			for(const MeshVoxelInfo& info : objectInfoData)
			{
				voxelInfo.push_back(
				{
					info.voxCount,
					info.voxOffset + voxelBatchOffset
				});
				voxelCount += info.voxCount;
			}
			voxelBatchOffset += voxelCount;
			assert(gfgFile.Header().meshes.front().headerCore.vertexCount == voxelCount);
			totalVoxelCount += voxelCount;

			// Now Load Actual Voxels		
			std::vector<uint8_t> voxelData(gfgFile.MeshVertexDataSize(0));
			gfgFile.MeshVertexData(voxelData.data(), 0);

			// Copy to Appropirate Buffers
			size_t internalOffset = 0;
			// Voxel Pos
			size_t oldSize = voxelPos.size();
			voxelPos.resize(voxelPos.size() + voxelCount);
			std::memcpy(voxelPos.data() + oldSize,
						voxelData.data() + internalOffset,
						voxelCount * sizeof(CVoxelPos));
			internalOffset += voxelCount * sizeof(CVoxelPos);
			// Voxel Norm
			oldSize = voxelNorm.size();
			voxelNorm.resize(voxelNorm.size() + voxelCount);
			std::memcpy(voxelNorm.data() + oldSize,
						voxelData.data() + internalOffset,
						voxelCount * sizeof(CVoxelNorm));
			internalOffset += voxelCount * sizeof(CVoxelNorm);
			// Voxel Albedo
			oldSize = voxelAlbedo.size();
			voxelNorm.resize(voxelAlbedo.size() + voxelCount);
			std::memcpy(voxelAlbedo.data() + oldSize,
						voxelData.data() + internalOffset,
						voxelCount * sizeof(CVoxelAlbedo));
			internalOffset += voxelCount * sizeof(CVoxelAlbedo);
			// Voxel Weights
			if(batch->MeshType() == MeshBatchType::SKELETAL)
			{
				oldSize = voxelWeights.size();
				voxelNorm.resize(voxelWeights.size() + voxelCount);
				std::memcpy(voxelWeights.data() + oldSize,
							voxelData.data() + internalOffset,
							voxelCount * sizeof(CVoxelWeights));
				internalOffset += voxelCount * sizeof(CVoxelWeights);
			}
			assert(voxelData.size() == internalOffset);
		}
	}

	// All Loaded to CPU
	// Now load it to GPU
	CascadeBatch cBatch = {};
	size_t newOffset = gpuBufferOffset;	
	// Mesh Voxel Info
	cBatch.dMeshVoxelInfo = reinterpret_cast<CMeshVoxelInfo*>(gpuData.Data() + newOffset);
	cudaMemcpy(gpuData.Data() + newOffset, voxelInfo.data(),
			   voxelInfo.size() * sizeof(MeshVoxelInfo), cudaMemcpyHostToDevice);
	newOffset += voxelInfo.size() * sizeof(MeshVoxelInfo);
	// Voxel Pos
	cBatch.dVoxelPos = reinterpret_cast<CVoxelPos*>(gpuData.Data() + newOffset);
	cudaMemcpy(gpuData.Data() + newOffset, voxelPos.data(),
			   voxelPos.size() * sizeof(CVoxelPos), cudaMemcpyHostToDevice);
	newOffset += voxelPos.size() * sizeof(CVoxelPos);
	// Voxel Norm
	cBatch.dVoxelNorm = reinterpret_cast<CVoxelNorm*>(gpuData.Data() + newOffset);
	cudaMemcpy(gpuData.Data() + newOffset, voxelNorm.data(),
			   voxelNorm.size() * sizeof(CVoxelNorm), cudaMemcpyHostToDevice);
	newOffset += voxelNorm.size() * sizeof(CVoxelNorm);
	// Voxel Albedo
	cBatch.dVoxelAlbedo = reinterpret_cast<CVoxelAlbedo*>(gpuData.Data() + newOffset);
	cudaMemcpy(gpuData.Data() + newOffset, voxelAlbedo.data(),
			   voxelAlbedo.size() * sizeof(CVoxelAlbedo), cudaMemcpyHostToDevice);
	newOffset += voxelAlbedo.size() * sizeof(CVoxelAlbedo);
	if(batch->MeshType() == MeshBatchType::SKELETAL)
	{
		cBatch.dVoxelWeight = reinterpret_cast<CVoxelWeights*>(gpuData.Data() + newOffset);
		cudaMemcpy(gpuData.Data() + newOffset, voxelWeights.data(),
				   voxelWeights.size() * sizeof(CVoxelWeights), cudaMemcpyHostToDevice);
		newOffset += voxelWeights.size() * sizeof(CVoxelWeights);
	}
	else cBatch.dVoxelWeight = nullptr;

	GI_LOG("\tCascade %f : %dvox", currentSpan, totalVoxelCount);
	cascadeDataPointers.push_back(cBatch);
	cascadeVoxelCount.push_back(totalVoxelCount);
	cascadeVoxelSizes.push_back(CaclulateCascadeMemoryUsageMB(totalVoxelCount, 
															  static_cast<uint32_t>(batch->DrawCount()),
															  batch->MeshType() == MeshBatchType::SKELETAL));
	return newOffset;
}

double VoxelCacheBatches::CaclulateCascadeMemoryUsageMB(uint32_t voxelCount, uint32_t meshCount, bool isSkeletal)
{
	size_t totalByteSize = voxelCount * (sizeof(CVoxelPos) +
										 sizeof(CVoxelNorm) +
										 sizeof(CVoxelAlbedo) +
										 (isSkeletal) ? sizeof(CVoxelWeights) : 0);

	return static_cast<float>(totalByteSize) / 1024.0 / 1024.0;
}

VoxelCacheBatches::VoxelCacheBatches()
	: batches(nullptr)
	, glAllocated(false)
{}

VoxelCacheBatches::VoxelCacheBatches(float minSpan, uint32_t levelCount,
									 const std::vector<MeshBatchI*>* batches,
									 const std::vector<std::vector<std::string>>& batchFileNames)
	: batches(batches)
	, glAllocated(false)
{
	IETimer t;
	t.Start();

	// Determine Total Size
	uint32_t i = 0;
	size_t totalSize = 0;
	float currentSpan = minSpan;
	for(const auto& fileNames : batchFileNames)
	{
		MeshBatchI * currentBatch = (*batches)[i];
		for(const auto& name : fileNames)
		{
			const auto voxelFile = GenVoxelGFGFileName(name, currentSpan);

			totalSize += FetchFileVoxelSize(voxelFile, currentBatch->MeshType() == MeshBatchType::SKELETAL,
											currentBatch->RepeatCount());
		}

		currentSpan *= 2.0f;
		i++;
	}

	// Allocate GPU
	gpuData.Resize(totalSize);

	// Now do the Loading
	i = 0;
	size_t currentOffset = 0;
	currentSpan = minSpan;
	for(const auto& fileNames : batchFileNames)
	{
		MeshBatchI * currentBatch = (*batches)[i];
		currentOffset = LoadBatchVoxels(currentOffset, currentSpan, currentBatch, fileNames);

		currentSpan *= 2.0f;
		i++;
	}
	t.Stop();
	GI_LOG("Voxel Caches Loaded. (%fms)", t.ElapsedMilliS());

	// Loaded
	//----------------------------

}

VoxelCacheBatches::VoxelCacheBatches(VoxelCacheBatches&&)
{

}


void VoxelCacheBatches::AllocateGL(uint32_t cascade)
{

}

void VoxelCacheBatches::DeallocateGL()
{

}

void VoxelCacheBatches::Draw(uint32_t cascade,
							 Shader& vDebugVoxel,
							 Shader& fDebugVoxel,
							 const Camera& camera)
{

}

const VoxelCacheBatches::CascadeBatch& VoxelCacheBatches::getCascade(uint32_t cascade, uint32_t batch) const
{
	return cascadeDataPointers[cascade * batches->size() + batch];
}