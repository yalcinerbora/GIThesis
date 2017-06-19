#include "IEUtility/IETimer.h"
#include "OGLTimer.h"
#include "GIVoxelCache.h"
#include "GFG/GFGFileLoader.h"
#include "Camera.h"
#include "MeshBatchSkeletal.h"
#include "Macros.h"
#include "GLSLBindPoints.h"
#include "DrawBuffer.h"

#include <sstream>
#include <cuda_gl_interop.h>
#include <numeric>

const std::string GIVoxelCache::GenVoxelGFGFileName(const std::string& fileName, float span)
{
	// Voxel File Name is "meshFileName"_vox_0.6.gfg
	size_t startPos = fileName.find_last_of("\\/");
	startPos = (startPos == std::string::npos) ? 0 : startPos + 1;
	size_t endPos = fileName.find_last_of(".");

	std::string fileNameOnly = fileName.substr(startPos, endPos);
	std::ostringstream voxPrefix;
	voxPrefix << fileNameOnly << "_vox_" << span << ".gfg";	
	return voxPrefix.str();
}

size_t GIVoxelCache::FetchFileVoxelSize(const std::string& voxelGFGFile,
										const MeshBatchI* batch)
{
	bool isSkeletal = batch->MeshType() == MeshBatchType::SKELETAL;

	std::ifstream stream(voxelGFGFile, std::ios_base::in | std::ios_base::binary);
	GFGFileReaderSTL stlFileReader(stream);
	GFGFileLoader gfgFile(&stlFileReader);

	// There are total of Two Meshes
	// Last gfg "mesh" holds object information
	// Other one holds voxels
	GFGFileError e = gfgFile.ValidateAndOpen();
	assert(e == GFGFileError::OK);
	const auto& header = gfgFile.Header();
	assert(gfgFile.Header().meshes.size() == 2);

	return gfgFile.MeshVertexDataSize(0) +
		   gfgFile.MeshVertexDataSize(1);
}

size_t GIVoxelCache::LoadBatchVoxels(size_t gpuBufferOffset, float currentSpan,
									 const MeshBatchI* batch,
									 const std::vector<std::string>& gfgFiles)
{
	if(batch->DrawCount() == 0)
	{
		hDeviceCascadePtrs.push_back({});
		cascadeVoxelCount.push_back(0);
		cascadeVoxelSizes.push_back(0);
		return gpuBufferOffset;
	}
	
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
			std::vector<MeshVoxelInfo> objectInfoData(gfgFile.MeshVertexDataSize(1) / sizeof(MeshVoxelInfo));
			gfgFile.MeshVertexData(reinterpret_cast<uint8_t*>(objectInfoData.data()), 1);

			// Some Validation
			const auto& component = meshObjCount.components[i];
			assert(component.dataType == GFGDataType::UINT32_2);
			assert(sizeof(MeshVoxelInfo) == GFGDataTypeByteSize[static_cast<int>(GFGDataType::UINT32_2)]);
			assert(component.internalOffset == 0);
			assert(component.logic == GFGVertexComponentLogic::POSITION);
			assert(component.stride == sizeof(MeshVoxelInfo));

			uint32_t voxelCount = 0;
			if(voxelBatchOffset != 0)
			{
				for(const MeshVoxelInfo& info : objectInfoData)
				{
					voxelInfo.push_back(
					{
						info.voxCount,
						info.voxOffset + voxelBatchOffset
					});
					voxelCount += info.voxCount;
				}
				assert(gfgFile.Header().meshes.front().headerCore.vertexCount == voxelCount);
			}
			else
			{
				voxelInfo.insert(voxelInfo.end(), objectInfoData.begin(), objectInfoData.end());
				voxelCount = static_cast<uint32_t>(gfgFile.Header().meshes.front().headerCore.vertexCount);				
			}
			voxelBatchOffset += voxelCount;
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
			voxelAlbedo.resize(voxelAlbedo.size() + voxelCount);
			std::memcpy(voxelAlbedo.data() + oldSize,
						voxelData.data() + internalOffset,
						voxelCount * sizeof(CVoxelAlbedo));
			internalOffset += voxelCount * sizeof(CVoxelAlbedo);
			// Voxel Weights
			if(batch->MeshType() == MeshBatchType::SKELETAL)
			{
				oldSize = voxelWeights.size();
				voxelWeights.resize(voxelWeights.size() + voxelCount);
				std::memcpy(voxelWeights.data() + oldSize,
							voxelData.data() + internalOffset,
							voxelCount * sizeof(CVoxelWeights));
				internalOffset += voxelCount * sizeof(CVoxelWeights);
			}
			assert(voxelData.size() == internalOffset);
		}
	}
	assert(voxelInfo.size() == batch->DrawCount());

	// All Loaded to CPU
	// Now load it to GPU
	BatchVoxelCache cBatch = {};
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
	hDeviceCascadePtrs.push_back(cBatch);
	cascadeVoxelCount.push_back(totalVoxelCount);
	cascadeVoxelSizes.push_back(CaclulateCascadeMemoryUsage(totalVoxelCount,
															static_cast<uint32_t>(batch->DrawCount()),
															batch->MeshType() == MeshBatchType::SKELETAL));
	return newOffset;
}

size_t GIVoxelCache::CaclulateCascadeMemoryUsage(uint32_t voxelCount,
												 uint32_t meshCount,
												 bool isSkeletal)
{
	size_t totalByteSize = voxelCount * (sizeof(CVoxelPos) +
										 sizeof(CVoxelNorm) +
										 sizeof(CVoxelAlbedo) +
										 ((isSkeletal) ? sizeof(CVoxelWeights) : 0));
	totalByteSize += meshCount * sizeof(MeshVoxelInfo);
	return totalByteSize;
}

GIVoxelCache::GIVoxelCache()
	: batches(nullptr)
	, baseSpan(0.0f)
	, currentCascade(-1)
{}

GIVoxelCache::GIVoxelCache(float baseSpan, uint32_t levelCount,
						   const std::vector<MeshBatchI*>* batches,
						   const std::vector<std::vector<std::string>>& batchFileNames)
	: batches(batches)
	, baseSpan(baseSpan)
	, vRenderVoxelSkel(ShaderType::VERTEX, "Shaders/VoxRenderSkeletal.vert")
	, vRenderVoxel(ShaderType::VERTEX, "Shaders/VoxRender.vert")
	, fRenderVoxel(ShaderType::FRAGMENT, "Shaders/VoxRender.frag")
	, currentCascade(-1)
{		
	GI_LOG("Loading Voxel Caches...");
	IETimer t;
	t.Start();

	// Determine Total Size
	size_t totalSize = 0;
	float currentSpan = baseSpan;
	for(uint32_t i = 0; i < levelCount; i++)
	{
		uint32_t batchId = 0;
		for(const auto& fileNames : batchFileNames)
		{
			MeshBatchI * currentBatch = (*batches)[batchId];
			for(const auto& name : fileNames)
			{
				const auto voxelFile = GenVoxelGFGFileName(name, currentSpan);
				totalSize += FetchFileVoxelSize(voxelFile, currentBatch);
			}
			batchId++;
		}
		currentSpan *= 2.0f;
	}

	// Allocate GPU
	gpuData.Resize(totalSize);

	// Now do the Loading
	size_t currentOffset = 0;
	currentSpan = baseSpan;
	for(uint32_t i = 0; i < levelCount; i++)
	{
		uint32_t batchId = 0;
		for(const auto& fileNames : batchFileNames)
		{
			MeshBatchI * currentBatch = (*batches)[batchId];
			currentOffset = LoadBatchVoxels(currentOffset, currentSpan, currentBatch, fileNames);
			batchId++;
		}
		currentSpan *= 2.0f;
	}
	t.Stop();
	assert(currentOffset == gpuData.Size());
	GI_LOG("Voxel Caches Loaded %.2fMB. (%fms)",
		   static_cast<double>(gpuData.Size()) / 1024.0 / 1024.0,
		   t.ElapsedMilliS());

	// Entire Cache System Loaded
	// Now load pointers to GPU
	dDeviceCascadePtrs = hDeviceCascadePtrs;
}

GIVoxelCache::GIVoxelCache(GIVoxelCache&& other)
	: batches(std::move(other.batches))
	, baseSpan(other.baseSpan)
	, gpuData(std::move(other.gpuData))
	, dDeviceCascadePtrs(std::move(other.dDeviceCascadePtrs))
	, hDeviceCascadePtrs(std::move(other.hDeviceCascadePtrs))
	, vRenderVoxelSkel(std::move(other.vRenderVoxelSkel))
	, vRenderVoxel(std::move(other.vRenderVoxel))
	, fRenderVoxel(std::move(other.fRenderVoxel))
	, currentCascade(other.currentCascade)
	, debugDrawBuffer(std::move(other.debugDrawBuffer))
	, meshData(std::move(other.meshData))
	, debugVAO(std::move(other.debugVAO))
	, cascadeVoxelCount(std::move(other.cascadeVoxelCount))
	, cascadeVoxelSizes(std::move(other.cascadeVoxelSizes))
{}

GIVoxelCache& GIVoxelCache::operator=(GIVoxelCache&& other)
{
	assert(&other != this);
	batches = std::move(other.batches);
	baseSpan = other.baseSpan;
	gpuData = std::move(other.gpuData);
	dDeviceCascadePtrs = std::move(other.dDeviceCascadePtrs);
	hDeviceCascadePtrs = std::move(other.hDeviceCascadePtrs);
	vRenderVoxelSkel = std::move(other.vRenderVoxelSkel);
	vRenderVoxel = std::move(other.vRenderVoxel);
	fRenderVoxel = std::move(other.fRenderVoxel);
	currentCascade = other.currentCascade;
	debugDrawBuffer = std::move(other.debugDrawBuffer);
	meshData = std::move(other.meshData);
	debugVAO = std::move(other.debugVAO);
	cascadeVoxelCount = std::move(other.cascadeVoxelCount);
	cascadeVoxelSizes = std::move(other.cascadeVoxelSizes);
	return *this;
}

void GIVoxelCache::AllocateGL(uint32_t cascade)
{
	if(currentCascade != static_cast<int32_t>(cascade))
	{
		DeallocateGL();
		VoxelVAO::CubeOGL cube = VoxelVAO::LoadCubeDataFromGFG();
		cubeIndexCount = cube.drawCount;

		size_t totalSize = std::accumulate(cascadeVoxelSizes.begin() + cascade * batches->size(),
										   cascadeVoxelSizes.begin() + (cascade + 1) * batches->size(),
										   size_t(0u));
		debugDrawBuffer.Resize(cube.data.size() + totalSize, false);

		cudaGraphicsResource_t glResource = nullptr;
		uint8_t* glCudaPointer = nullptr;
		size_t bufferSize = 0;

		CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&glResource, debugDrawBuffer.getGLBuffer(),
												cudaGraphicsMapFlagsWriteDiscard));
		CUDA_CHECK(cudaGraphicsMapResources(1, &glResource));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&glCudaPointer),
														&bufferSize,
														glResource));
		assert(bufferSize == debugDrawBuffer.Capacity());

		// Mapped now just do a memcpy
		// One for cube indices/vertices and one for voxel data
		size_t readOffset = std::accumulate(cascadeVoxelSizes.begin(),
											cascadeVoxelSizes.begin() + cascade * batches->size(),
											size_t(0u));
		const uint8_t* readPointer = gpuData.Data() + readOffset;
		CUDA_CHECK(cudaMemcpy(glCudaPointer, 
							  cube.data.data(),
							  cube.data.size(),
							  cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(glCudaPointer + cube.data.size(),
							  readPointer, 
							  bufferSize - cube.data.size(),
							  cudaMemcpyDeviceToDevice));
		// Unmap and unregister we are done
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &glResource));
		CUDA_CHECK(cudaGraphicsUnregisterResource(glResource));

		size_t cubeOffset = cube.data.size();
		size_t cubeVertexOffset = cube.drawCount * sizeof(uint32_t);
		size_t offset = cubeOffset;
		for(size_t i = 0; i < batches->size(); i++)
		{
			if((*batches)[i]->DrawCount() == 0)
			{
				meshData.emplace_back();
				debugVAO.emplace_back();
			}
			else
			{

				// Generate VAO
				const BatchVoxelCache& cc = hDeviceCascadePtrs[cascade * batches->size() + i];
				const uint32_t voxelCount = cascadeVoxelCount[cascade * batches->size() + i];

				offset += (*batches)[i]->DrawCount() * sizeof(MeshVoxelInfo);
				size_t vPosOffset = offset;
				offset += voxelCount * sizeof(VoxelPosition);
				size_t vNormOffset = offset;
				offset += voxelCount * sizeof(VoxelNormal);
				size_t vAlbedoOffset = offset;
				offset += voxelCount * sizeof(VoxelAlbedo);
				size_t vWeightOffset = 0;
				if(cc.dVoxelWeight != nullptr)
				{
					vWeightOffset = offset;
					offset += voxelCount * sizeof(VoxelWeights);
				}

				debugVAO.emplace_back(debugDrawBuffer,
									  cubeVertexOffset,
									  vPosOffset,
									  vNormOffset,
									  vAlbedoOffset,
									  vWeightOffset);

				// Memcopy cascade voxel sizes to CPU (in order to launch draw calls)
				meshData.emplace_back(CopyMeshObjectInfo(cascade, static_cast<uint32_t>(i)));
			}
		}
		assert(offset == debugDrawBuffer.Capacity());
		currentCascade = cascade;
	}
}

void GIVoxelCache::DeallocateGL()
{
	if(currentCascade != -1)
	{
		debugVAO.clear();
		meshData = std::move(std::vector<std::vector<MeshVoxelInfo>>());
		debugDrawBuffer = std::move(StructuredBuffer<uint8_t>());		
		currentCascade = -1;
	}
}

double GIVoxelCache::Draw(bool doTiming, 
						  const Camera& camera,
						  VoxelRenderType renderType)
{
	// Timing
	OGLTimer t;
	if(doTiming) t.Start();
	
	// Framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(camera.width),
			   static_cast<GLsizei>(camera.height));
	
	// State
	glDisable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LEQUAL);
	glDepthMask(true);
	glColorMask(true, true, true, true);	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	fRenderVoxel.Bind();
			
	// Buffers
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	for(unsigned int i = 0; i < batches->size(); i++)
	{		
		if((*batches)[i]->DrawCount() == 0) continue;

		(*batches)[i]->getDrawBuffer().BindAABB(LU_AABB);
		(*batches)[i]->getDrawBuffer().BindModelTransform(LU_MTRANSFORM);
		(*batches)[i]->getDrawBuffer().BindModelTransformIndex(LU_MTRANSFORM_INDEX);

		if((*batches)[i]->MeshType() == MeshBatchType::SKELETAL)
		{
			// Shader Vert
			vRenderVoxelSkel.Bind();
			glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(renderType));
			glUniform1f(U_SPAN, baseSpan * static_cast<float>(1 << currentCascade));
	
			// Joint Transforms
			MeshBatchSkeletal* batchPtr = static_cast<MeshBatchSkeletal*>((*batches)[i]);
			batchPtr->getJointTransforms().BindAsShaderStorageBuffer(LU_JOINT_TRANS);
		}
		else
		{
			// Shader Vert
			vRenderVoxel.Bind();
			glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(renderType));
			glUniform1f(U_SPAN, baseSpan * static_cast<float>(1 << currentCascade));
		}

		debugVAO[i].Bind();
		const auto& batchMeshInfo = meshData[i];
		for(GLuint drawId = 0; drawId < batchMeshInfo.size(); drawId++)
		{
			const auto& meshInfo = batchMeshInfo[drawId];
			if(meshInfo.voxCount == 0) continue;
			glUniform1ui(U_DRAW_ID, drawId);
			debugVAO[i].Draw(cubeIndexCount,
							 meshInfo.voxCount,
							 meshInfo.voxOffset);
		}
	}

	// Timer
	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMS();
	}
	return t.ElapsedMS();
}

const std::vector<CMeshVoxelInfo> GIVoxelCache::CopyMeshObjectInfo(uint32_t cascadeId, uint32_t batchId) const
{
	std::vector<CMeshVoxelInfo> mesh((*batches)[batchId]->DrawCount());
	const BatchVoxelCache& cc = hDeviceCascadePtrs[cascadeId * batches->size() + batchId];
	CUDA_CHECK(cudaMemcpy(mesh.data(),
						  cc.dMeshVoxelInfo,
						  (*batches)[batchId]->DrawCount() * sizeof(MeshVoxelInfo),
						  cudaMemcpyDeviceToHost));
	return mesh;
}

const std::vector<BatchVoxelCache>& GIVoxelCache::getDeviceCascadePointersHost() const
{
	return hDeviceCascadePtrs;
}

const CudaVector<BatchVoxelCache>& GIVoxelCache::getDeviceCascadePointersDevice() const
{
	return dDeviceCascadePtrs;
}