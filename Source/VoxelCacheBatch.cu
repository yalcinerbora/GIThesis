#include "VoxelCacheBatch.h"
#include "GFG/GFGFileLoader.h"
#include "MeshBatchI.h"
#include <sstream>

const std::string VoxelCacheBatch::GenVoxelGFGFileName(const std::string& fileName, float span)
{
	// Voxel File Name is "meshFileName"_0.6_vox.gfg
	size_t startPos = fileName.find_last_of("\\/");
	startPos = (startPos == std::string::npos) ? 0 : startPos;
	size_t endPos = fileName.find_last_of(".");

	std::string fileNameOnly = fileName.substr(startPos + 1, endPos);
	std::ostringstream voxPrefix;
	voxPrefix << fileNameOnly << "_" << span << "_vox.gfg";
	
	return voxPrefix.str();
}

uint32_t VoxelCacheBatch::FetchVoxelCount(const std::string& voxelGFGFile)
{
	return 0;
}

size_t VoxelCacheBatch::LoadVoxels(size_t offset, int repeatCount,
								   const std::string& voxelGFGFile)
{
	return 0;
	//std::ifstream stream(voxelGFGFile, std::ios_base::in | std::ios_base::binary);
	//GFGFileReaderSTL stlFileReader(stream);
	//GFGFileLoader gfgFile(&stlFileReader);

	//// There are total of Two Meshes
	//// Last gfg "mesh" holds object information
	//// Other one holds voxels
	//GFGFileError e = gfgFile.ValidateAndOpen();
	//const auto& header = gfgFile.Header();
	//assert(e == GFGFileError::OK);
	//assert(gfgFile.Header()eader.meshes.size() == 2);

	//// First mesh contains objInfos
	//const auto& meshObjCount = header.meshes.back();


	//uint32_t objCount = static_cast<uint32_t>(meshObjCount.headerCore.vertexCount);
	//std::vector<uint8_t> objectInfoData(gfgFile.MeshVertexDataSize(1));
	//gfgFile.MeshVertexData(objectInfoData.data(), 1);

	//// Determine VoxelCount
	//for(uint32_t i = 0; i < cascadeCount; i++)
	//{
	//	const auto& mesh = header.meshes[i];

	//	// Special case aabbmin show span count
	//	assert(scenes[i].span == mesh.headerCore.aabb.min[0]);
	//	scenes[i].cache.emplace_back(mesh.headerCore.vertexCount * repeatCount, objCount * repeatCount, isSkeletal);

	//	// Load to Mem
	//	std::vector<uint8_t> meshData(gfgFile.MeshVertexDataSize(i));
	//	gfgFile.MeshVertexData(meshData.data(), i);

	//	auto& currentCache = scenes[i].cache.back();

	//	// Object gridInfo
	//	const auto& component = meshObjCount.components[i];
	//	assert(component.dataType == GFGDataType::UINT32_2);
	//	assert(sizeof(ObjGridInfo) == GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)]);
	//	assert(component.internalOffset == 0);
	//	assert(component.logic == GFGVertexComponentLogic::POSITION);
	//	assert(component.stride == sizeof(ObjGridInfo));

	//	currentCache.objInfo.CPUData().resize(objCount * repeatCount);
 //       for(int j = 0; j < repeatCount; j++)
 //       {
 //           std::memcpy(currentCache.objInfo.CPUData().data() + objCount * j,
 //                       objectInfoData.data() + component.startOffset,
 //                       objCount * component.stride);
 //       }

	//	// Voxel Data
	//	for(const auto& component : mesh.components)
	//	{
	//		if(component.logic == GFGVertexComponentLogic::POSITION)
	//		{
	//			// NormPos
	//			assert(component.dataType == GFGDataType::UINT32_2);
	//			auto& normPosVector = currentCache.voxelNormPos.CPUData();

	//			normPosVector.resize(mesh.headerCore.vertexCount * repeatCount);
 //               for(int j = 0; j < repeatCount; j++)
 //               {
 //                   std::memcpy(normPosVector.data() + mesh.headerCore.vertexCount * j,
 //                               meshData.data() + component.startOffset,
 //                               mesh.headerCore.vertexCount * component.stride);
 //               }
	//		}
	//		else if(component.logic == GFGVertexComponentLogic::NORMAL)
	//		{
	//			// Vox Ids
	//			assert(component.dataType == GFGDataType::UINT32_2);
	//			auto& voxIdsVector = currentCache.voxelIds.CPUData();

	//			voxIdsVector.resize(mesh.headerCore.vertexCount * repeatCount);
 //               for(int j = 0; j < repeatCount; j++)
 //               {
 //                   std::memcpy(voxIdsVector.data() + mesh.headerCore.vertexCount * j,
 //                               meshData.data() + component.startOffset,
 //                               mesh.headerCore.vertexCount * component.stride);
 //               }
	//		}
	//		else if(component.logic == GFGVertexComponentLogic::COLOR)
	//		{
	//			// Color
	//			assert(component.dataType == GFGDataType::UNORM8_4);
	//			auto& voxColorVector = currentCache.voxelRenderData.CPUData();

	//			voxColorVector.resize(mesh.headerCore.vertexCount * repeatCount);
 //               for(int j = 0; j < repeatCount; j++)
 //               {
 //                   std::memcpy(voxColorVector.data() + mesh.headerCore.vertexCount * j,
 //                               meshData.data() + component.startOffset,
 //                               mesh.headerCore.vertexCount * component.stride);
 //               }
	//		}
	//		else if(component.logic == GFGVertexComponentLogic::WEIGHT)
	//		{
	//			// Weight
	//			assert(component.dataType == GFGDataType::UINT32_2);
	//			auto& voxWeightVector = currentCache.voxelWeightData.CPUData();

	//			voxWeightVector.resize(mesh.headerCore.vertexCount * repeatCount);
 //               for(int j = 0; j < repeatCount; j++)
 //               {
 //                   std::memcpy(voxWeightVector.data() + mesh.headerCore.vertexCount * j,
 //                               meshData.data() + component.startOffset,
 //                               mesh.headerCore.vertexCount * component.stride);
 //               }
	//		}
	//		else
	//		{
	//			assert(false);
	//		}
	//		currentCache.voxelRenderData.SendData();
	//		currentCache.voxelNormPos.SendData();
	//		currentCache.voxelIds.SendData();
	//		currentCache.objInfo.SendData();
	//		if(isSkeletal) currentCache.voxelWeightData.SendData();
	//	}

	//	GI_LOG("\tCascade#%d Voxels : %zd", i, mesh.headerCore.vertexCount  * repeatCount);
	//}
	//return true;
}

double VoxelCacheBatch::CaclulateCascadeMemoryUsageMB(uint32_t cascade) const
{
	return 0.0f;
	//static_cast<double>(gpuData.Size() + debugDrawBuffer.Count()) / 1024.0 / 1024.0;
}

VoxelCacheBatch::VoxelCacheBatch()
	: batch(nullptr)
	, isSkeletal(false)
	, glAllocated(false)
{}

VoxelCacheBatch::VoxelCacheBatch(float minSpan, uint32_t levelCount, MeshBatchI& batch,								 
								 const std::vector<std::string>& batchNames)
	: batch(&batch)
	, isSkeletal(batch.MeshType() == MeshBatchType::SKELETAL)
	, glAllocated(false)
{

	//double ThesisSolution::LoadBatchVoxels(MeshBatchI* batch)
	//{
	//	//IETimer t;
	//	//t.Start();
	//
	//	//// Voxelization
	//	//std::stringstream voxPrefix;
	//	//voxPrefix << "vox_" << CascadeSpan << "_" << GI_CASCADE_COUNT << "_";
	//	//
	//	//// Load GFG
	//	//std::string batchVoxFile = voxPrefix.str() + batch->BatchName() + ".gfg";
	//	//LoadVoxel(voxelCaches, batchVoxFile.c_str(), GI_CASCADE_COUNT,
	//	//		  batch->MeshType() == MeshBatchType::RIGID,
	// //             batch->RepeatCount());
	//
	//	//t.Stop();
	//	//// Voxel Load Complete
	//	//GI_LOG("Loading \"%s\" complete", batchVoxFile.c_str());
	//	//GI_LOG("\tDuration : %f ms", t.ElapsedMilliS());
	//	//GI_LOG("------");
	//	//return t.ElapsedMilliS();
	//	return 0.0;
	//}

	// Determine Size
	for(const std::string& f : batchNames)
	{

	}
//	batch.
}

VoxelCacheBatch::VoxelCacheBatch(VoxelCacheBatch&&)
{

}


void VoxelCacheBatch::AllocateGL(uint32_t cascade)
{

}

void VoxelCacheBatch::DeallocateGL()
{

}

void VoxelCacheBatch::Draw(uint32_t cascade,
						   Shader& vDebugVoxel,
						   Shader& fDebugVoxel,
						   const Camera& camera)
{

}

const VoxelCacheBatch::Cascade& VoxelCacheBatch::getCascade(uint32_t cascade) const
{
	return cascadeDataPointers[cascade];
}