#include "GIVoxelPages.h"
#include "PageKernels.cuh"
#include "DrawBuffer.h"
#include "CudaInit.h"
#include "CudaTimer.h"
#include "GIVoxelCache.h"
#include "GISparseVoxelOctree.h"
#include "MeshBatchSkeletal.h"
#include "OGLTimer.h"
#include "IEUtility/IEMath.h"
#include "GLSLBindPoints.h"
#include "Camera.h"
#include <cuda_gl_interop.h>
#include "IEUtility/IEAxisAalignedBB.h"

inline static std::ostream& operator<<(std::ostream& ostr, const CSegmentInfo& segObj)
{
	uint16_t cascadeNo = (segObj.packed >> 14) & 0x0003;
	uint16_t objType = (segObj.packed >> 12) & 0x0003;
	uint16_t occupation = (segObj.packed >> 10) & 0x0003;

	ostr << cascadeNo << ", ";
	ostr << segObj.batchId << ", ";
	ostr << segObj.objId << " | ";

	ostr << segObj.objectSegmentId << " | ";
	ostr << objType << " | ";
	ostr << occupation << " | ";
	return ostr;
}

GIVoxelPages::FastVoxelizer::FastVoxelizer()
	: denseResource(nullptr)
	, octreeParams(nullptr)
{}

GIVoxelPages::FastVoxelizer::FastVoxelizer(OctreeParameters* octreeParams)
	: denseResource(nullptr)
	, octreeParams(octreeParams)
{
	size_t offset = 0;
	
	// Grid Transform
	offset += DeviceOGLParameters::UBOAlignOffset(offset);
	gridTransformOffset = static_cast<GLuint>(offset);
	offset += sizeof(FrameTransformData);
	// Dense
	offset += DeviceOGLParameters::SSBOAlignOffset(offset);
	denseOffset = static_cast<GLuint>(offset);
	offset += octreeParams->CascadeBaseLevelSize *
			  octreeParams->CascadeBaseLevelSize *
			  octreeParams->CascadeBaseLevelSize * sizeof(uint2);
	// Allocator
	allocatorOffset = static_cast<GLuint>(offset);
	offset += sizeof(uint32_t);

	// Gen OGL Buffers
	oglData.Resize(offset, false);
	
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&denseResource,
											oglData.getGLBuffer(),
											cudaGraphicsRegisterFlagsReadOnly));
}

GIVoxelPages::FastVoxelizer::FastVoxelizer(FastVoxelizer&& other)
	: oglData(std::move(other.oglData))
	, denseResource(other.denseResource)
	, octreeParams(other.octreeParams)
{
	other.denseResource = nullptr;
}

GIVoxelPages::FastVoxelizer& GIVoxelPages::FastVoxelizer::operator=(FastVoxelizer&& other)
{
	assert(this != &other);
	if(denseResource) CUDA_CHECK(cudaGraphicsUnregisterResource(denseResource));
	oglData = std::move(other.oglData);
	denseResource = other.denseResource;
	octreeParams = other.octreeParams;
	other.denseResource = nullptr;
	return *this;
}

GIVoxelPages::FastVoxelizer::~FastVoxelizer()
{
	if(denseResource) CUDA_CHECK(cudaGraphicsUnregisterResource(denseResource));
}

double GIVoxelPages::FastVoxelizer::Voxelize(const std::vector<MeshBatchI*>& batches,
										   const IEVector3& gridCenter,
										   const IEVector3& gridCorner,
										   bool doTiming)
{
	OGLTimer t;
	if(doTiming) t.Start();

	// States
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glEnable(GL_MULTISAMPLE);
	//glEnable(GL_CONSERVATIVE_RASTERIZATION_NV);
	glDepthMask(false);
	glStencilMask(0x0000);
	glColorMask(false, false, false, false);

	GLsizei totalSize = static_cast<GLsizei>(octreeParams->CascadeBaseLevelSize);
	glViewport(0, 0, totalSize, totalSize);

	// Generate GridAABB
	float worldWidthHeight = gridCenter[0] - gridCorner[0];
	IEVector3 gridMin = gridCorner;
	IEVector3 gridMax = gridMin + (worldWidthHeight * 2.0f);
	IEAxisAlignedBB3 gridAABB(gridMin, gridMax);

	// Generate Ortho Projection and View
	IEMatrix4x4 projectionAndView[2];
	projectionAndView[0] = IEMatrix4x4::Ortogonal(worldWidthHeight, worldWidthHeight,
												  worldWidthHeight * 0.5f,
												  worldWidthHeight * 0.5f);
	projectionAndView[1] = IEMatrix4x4::LookAt(gridCenter,
											   gridCenter - IEVector3::ZAxis,
											   IEVector3::YAxis);
	oglData.SendSubData(reinterpret_cast<const uint8_t*>(projectionAndView),
						gridTransformOffset,
						sizeof(FrameTransformData));

	
	// Dense Buffer & GridTransform buffer
	oglData.BindAsUniformBuffer(U_GRID_TRANSFORM, gridTransformOffset, sizeof(FrameTransformData));
	oglData.BindAsShaderStorageBuffer(LU_VOXEL_RENDER, denseOffset,
									  octreeParams->CascadeBaseLevelSize *
									  octreeParams->CascadeBaseLevelSize *
									  octreeParams->CascadeBaseLevelSize);

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	fragVoxelizeFast.Bind();

	for(MeshBatchI* batch : batches)
	{
		DrawBuffer& drawBuffer = batch->getDrawBuffer();
		VertexBuffer& vertexBuffer = batch->getVertexBuffer();

		// Batch Binds
		vertexBuffer.Bind();
		drawBuffer.BindModelTransform(LU_MTRANSFORM);
		drawBuffer.BindAsDrawIndirectBuffer();
		if(batch->MeshType() == MeshBatchType::SKELETAL)
		{
			MeshBatchSkeletal* batchPtr = static_cast<MeshBatchSkeletal*>(batch);
			batchPtr->getJointTransforms().BindAsShaderStorageBuffer(LU_JOINT_TRANS);
			vertVoxelizeFastSkeletal.Bind();
		}
		else vertVoxelizeFast.Bind();
		
		// For each object
		for(uint32_t drawId = 0; drawId < batch->DrawCount(); drawId++)
		{
			// Do a AABB check with grid and skip if out of bounds
			const auto& aabbData = drawBuffer.getAABB(drawId);
			IEAxisAlignedBB3 objectAABB(aabbData.min, aabbData.max);
			if(!objectAABB.Intersects(gridAABB)) continue;
			
			// Bind material and draw
			drawBuffer.BindMaterialForDraw(drawId);
			drawBuffer.DrawCallSingle(drawId);
		}		
	}

	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMS();
	}
	return 0.0;
}

double GIVoxelPages::FastVoxelizer::Filter(uint32_t& offset, CVoxelPage* dVoxelPages,
										   uint32_t pageCapacity, uint32_t cascadeId,
										   bool doTiming)
{
	CudaTimer t;
	if(doTiming) t.Start();

	// Map to CUDA
	uint8_t* oglDataCUDA; size_t size = 0;
	CUDA_CHECK(cudaGraphicsMapResources(1, &denseResource));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&oglDataCUDA),
													&size, denseResource));
	uint2* dDenseData = reinterpret_cast<uint2*>(oglDataCUDA + denseOffset);
	uint32_t& dAllocator = reinterpret_cast<uint32_t&>(oglDataCUDA[allocatorOffset]);

	// Filter valid voxels to page system
	int totalSize = octreeParams->CascadeBaseLevelSize *
					octreeParams->CascadeBaseLevelSize *
					octreeParams->CascadeBaseLevelSize;
	int gridSize = CudaInit::GenBlockSize(totalSize);
	int blockSize = CudaInit::TBP;

	// KC
	FilterVoxels<<<gridSize, blockSize>>>(// Voxel System
										  dVoxelPages,
										  // Dense Data from OGL
										  dAllocator,
										  dDenseData,
										  // Limits
										  cascadeId,
										  offset);

	// Fetch Allocator for total voxel count
	// then determine offset
	//offset += ??;

	// Unmap
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &denseResource));

	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMilliS();
	}
	return 0.0;
}

double GIVoxelPages::FastVoxelizer::FastVoxelize(CVoxelPage* dVoxelPages, uint32_t pageCount,
												 const CVoxelGrid* dVoxelGrids, uint32_t gridCount,
												 const std::vector<MeshBatchI*>& batches,
												 bool doTiming)
{
	assert(denseResource);

	// Copy new positions
	std::vector<IEVector3> gridPos(gridCount, IEVector3::ZeroVector);
	CUDA_CHECK(cudaMemcpy2D(gridPos.data(), sizeof(IEVector3),
							dVoxelPages, sizeof(CVoxelGrid),
							sizeof(IEVector3), gridCount,
							cudaMemcpyDeviceToHost));

	double totalTime = 0.0f;

	// Find Center Pos
	uint32_t usedSegmentCount = 0;
	for(uint32_t i = 0; i < gridCount; i++)
	{
		// Clear oglData
		oglData.Memset(static_cast<uint32_t>(0x0));

		// Voxelize to Center Grid
		IEVector3 gridCenter = gridPos[i] + (octreeParams->BaseSpan *
											 octreeParams->CascadeBaseLevel *
											 (0x1 << (gridCount - i - 1)));
		totalTime += Voxelize(batches, gridCenter, gridPos[i], doTiming);

		// Filter valid voxel to page system
		totalTime += Filter(usedSegmentCount, dVoxelPages, pageCount, i, doTiming);
	}
	return totalTime;
}

GIVoxelPages::PageRenderer::PageRenderer()
	: debugBufferResource(nullptr)
	, debugBufferCUDA(nullptr)
	, drawParameterOffset(0)
	, atomicIndexOffset(0)
	, gridInfoOffset(0)
	, voxelPositionOffset(0)
	, voxelRenderOffset(0)
{}

GIVoxelPages::PageRenderer::PageRenderer(const GIVoxelPages& pages)
	: vRenderWorldVoxel(ShaderType::VERTEX, "Shaders/VoxRenderWorld.vert")
	, fRenderWorldVoxel(ShaderType::FRAGMENT, "Shaders/VoxRender.frag")
	, debugBufferResource(nullptr)
	, debugBufferCUDA(nullptr)
	, drawParameterOffset(0)
	, atomicIndexOffset(0)
	, gridInfoOffset(0)
	, voxelPositionOffset(0)
	, voxelRenderOffset(0)
{
	VoxelVAO::CubeOGL cube = VoxelVAO::LoadCubeDataFromGFG();
	size_t maxVoxelCount = pages.dPages.Size() * PageSize;

	// Since Grid info will be bound as SSBO it inneds to be properly aligned
	size_t cubeOffset = cube.data.size();
	size_t cubeVertexOffset = cube.drawCount * sizeof(uint32_t);

	// Grid Info
	size_t offset = cubeOffset;
	offset = DeviceOGLParameters::SSBOAlignOffset(offset);
	gridInfoOffset = offset;
	offset += pages.svoParams->CascadeCount * sizeof(CVoxelGrid);
	
	// Atomic Index
	drawParameterOffset = offset;
	atomicIndexOffset = offset + offsetof(DrawPointIndexed, instanceCount);
	offset += sizeof(DrawPointIndexed);
	// Voxel Positions
	voxelPositionOffset = offset;
	offset += maxVoxelCount * sizeof(VoxelPosition);
	// Voxel Albedo or Normal
	voxelRenderOffset = offset;
	offset += maxVoxelCount * sizeof(VoxelNormal);
	static_assert(sizeof(VoxelNormal) == sizeof(VoxelAlbedo), "Implementation assumes all debug render types has the same size");

	// Allocate
	debugDrawBuffer.Resize(offset, false);

	// Now Register
	size_t bufferSize = 0;
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&debugBufferResource, debugDrawBuffer.getGLBuffer(),
											cudaGraphicsMapFlagsWriteDiscard));
	CUDA_CHECK(cudaGraphicsMapResources(1, &debugBufferResource));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&debugBufferCUDA),
													&bufferSize,
													debugBufferResource));
	assert(bufferSize == debugDrawBuffer.Capacity());

	// Copy Cube Vertex and Indices
	CUDA_CHECK(cudaMemcpy(debugBufferCUDA,
						  cube.data.data(),
						  cube.data.size(),
						  cudaMemcpyHostToDevice));

	// Copy Grid Info
	CUDA_CHECK(cudaMemcpy(debugBufferCUDA + gridInfoOffset,
						  pages.dVoxelGrids,
						  pages.svoParams->CascadeCount * sizeof(CVoxelGrid),
						  cudaMemcpyDeviceToDevice));

	// Copy Draw Point
	DrawPointIndexed dp =
	{
		cube.drawCount,
		0,	// Instance count will be filled each frame
		0,
		0,
		0
	};
	CUDA_CHECK(cudaMemcpy(debugBufferCUDA + drawParameterOffset,
						  &dp, sizeof(DrawPointIndexed),
						  cudaMemcpyHostToDevice));

	// All Done! (Unmap and continue)
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &debugBufferResource));
	debugBufferCUDA = nullptr;

	// Finally Generate VAO
	debugDrawVao = VoxelVAO(debugDrawBuffer,
							cubeVertexOffset,
							voxelPositionOffset,
							voxelRenderOffset);
}

GIVoxelPages::PageRenderer::PageRenderer(PageRenderer&& other)
	: vRenderWorldVoxel(std::move(other.vRenderWorldVoxel))
	, fRenderWorldVoxel(std::move(other.fRenderWorldVoxel))
	, debugBufferResource(other.debugBufferResource)
	, debugDrawBuffer(std::move(other.debugDrawBuffer))
	, debugBufferCUDA(other.debugBufferCUDA)
	, debugDrawVao(std::move(other.debugDrawVao))
	, drawParameterOffset(other.drawParameterOffset)
	, atomicIndexOffset(other.atomicIndexOffset)
	, gridInfoOffset(other.gridInfoOffset)
	, voxelPositionOffset(other.voxelPositionOffset)
	, voxelRenderOffset(other.voxelRenderOffset)
{
	other.debugBufferResource = nullptr;
	other.debugBufferCUDA = nullptr;
}

GIVoxelPages::PageRenderer& GIVoxelPages::PageRenderer::operator=(PageRenderer&& other)
{
	if(debugBufferResource)
		CUDA_CHECK(cudaGraphicsUnregisterResource(debugBufferResource));

	vRenderWorldVoxel = std::move(other.vRenderWorldVoxel);
	fRenderWorldVoxel = std::move(other.fRenderWorldVoxel);
	debugBufferResource = other.debugBufferResource;
	debugDrawBuffer = std::move(other.debugDrawBuffer);
	debugBufferCUDA = other.debugBufferCUDA;
	debugDrawVao = std::move(other.debugDrawVao);
	drawParameterOffset = other.drawParameterOffset;
	atomicIndexOffset = other.atomicIndexOffset;
	gridInfoOffset = other.gridInfoOffset;
	voxelPositionOffset = other.voxelPositionOffset;
	voxelRenderOffset = other.voxelRenderOffset;

	other.debugBufferResource = nullptr;
	other.debugBufferCUDA = nullptr;
	return *this;
}

GIVoxelPages::PageRenderer::~PageRenderer()
{
	if(debugBufferResource)
		CUDA_CHECK(cudaGraphicsUnregisterResource(debugBufferResource));
}

double GIVoxelPages::PageRenderer::Draw(bool doTiming,
										uint32_t cascade,
										VoxelRenderType renderType,
										const Camera& camera,
										const GIVoxelCache& cache,
										const GIVoxelPages& pages)
{
	// Skip if not allocated
	if(!Allocated()) return 0.0;

	CudaTimer cT;
	if(doTiming) cT.Start();

	// Map Buffer
	size_t bufferSize = 0;
	CUDA_CHECK(cudaGraphicsMapResources(1, &debugBufferResource));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&debugBufferCUDA),
													&bufferSize,
													debugBufferResource));
	assert(bufferSize == debugDrawBuffer.Capacity());

	// Copy Requested Data
	// Gen pointers
	VoxelPosition* voxelPosition = reinterpret_cast<VoxelPosition*>(debugBufferCUDA + voxelPositionOffset);
	unsigned int* voxelRender = reinterpret_cast<unsigned int*>(debugBufferCUDA + voxelRenderOffset);
	unsigned int* atomicIndex = reinterpret_cast<unsigned int*>(debugBufferCUDA + atomicIndexOffset);

	// Clear atomic counter
	CUDA_CHECK(cudaMemset(atomicIndex, 0x00, sizeof(unsigned int)));

	// Load new Grid Positions
	// Copy Grid Info
	CUDA_CHECK(cudaMemcpy2D(debugBufferCUDA + gridInfoOffset, sizeof(CVoxelGrid),
							pages.dVoxelGrids, sizeof(CVoxelGrid),
							sizeof(float3), pages.svoParams->CascadeCount,
							cudaMemcpyDeviceToDevice));

	// KC
	int gridSize = CudaInit::GenBlockSize(static_cast<int>(pages.dPages.Size() * PageSize));
	int blockSize = CudaInit::TBP;
	CopyPage<<<gridSize, blockSize>>>(// OGL Buffer
								      voxelPosition,
								      voxelRender,
								      *atomicIndex,
								      // Voxel Cache
								      cache.getDeviceCascadePointersDevice().Data(),
								      // Voxel Pages
								      reinterpret_cast<const CVoxelPageConst*>(pages.dPages.Data()),
								      //
								      static_cast<uint32_t>(pages.batches->size()),
								      cascade,
								      renderType);
	CUDA_KERNEL_CHECK();

	//// DEBUG
	//uint32_t nodesInCirculation = 0;
	//CUDA_CHECK(cudaMemcpy(&nodesInCirculation, atomicIndex, sizeof(uint32_t), cudaMemcpyDeviceToHost));
	//GI_LOG("Total Valid node count in pages : %d", nodesInCirculation);

	// Unmap buffer and continue
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &debugBufferResource));
	debugBufferCUDA = nullptr;

	// Timing
	OGLTimer t;
	if(doTiming)
	{
		cT.Stop();
		t.Start();
	}
	
	// Now render
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
	vRenderWorldVoxel.Bind();
	glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(renderType));
	fRenderWorldVoxel.Bind();

	// Uniforms
	debugDrawBuffer.BindAsShaderStorageBuffer(LU_VOXEL_GRID_INFO, 
											  static_cast<GLuint>(gridInfoOffset), 
											  static_cast<GLuint>(pages.svoParams->CascadeCount * sizeof(CVoxelGrid)));

	// Draw Indirect Buffer
	debugDrawBuffer.BindAsDrawIndirectBuffer();

	// VAO
	debugDrawVao.Bind();
	debugDrawVao.Draw(static_cast<GLuint>(drawParameterOffset));

	// Timer
	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMS() + cT.ElapsedMilliS();
	}
	return 0.0;
}

bool GIVoxelPages::PageRenderer::Allocated() const
{
	return vRenderWorldVoxel.IsValid();
}

GIVoxelPages::MultiPage::MultiPage(size_t pageCount)
{
	assert(pageCount != 0);
	size_t sizePerPage = GIVoxelPages::PageSize *
						 (sizeof(CVoxelPos) +
						  sizeof(CVoxelNorm) +
						  sizeof(CVoxelOccupancy))
						 +
						 GIVoxelPages::SegmentSize *
						 (sizeof(unsigned char) +
						  sizeof(CSegmentInfo));

	size_t totalSize = sizePerPage * pageCount;
	pageData.Resize(totalSize);
	pageData.Memset(0x00, 0, totalSize);
	
	uint8_t* dPtr = pageData.Data();
	ptrdiff_t offset = 0;
	for(size_t i = 0; i < pageCount; i++)
	{
		CVoxelPage page = {};

		page.dGridVoxPos = reinterpret_cast<CVoxelPos*>(dPtr + offset);
		offset += GIVoxelPages::PageSize * sizeof(CVoxelPos);

		page.dGridVoxNorm = reinterpret_cast<CVoxelNorm*>(dPtr + offset);
		offset += GIVoxelPages::PageSize * sizeof(CVoxelNorm);
		
		page.dGridVoxOccupancy = reinterpret_cast<CVoxelOccupancy*>(dPtr + offset);
		offset += GIVoxelPages::PageSize * sizeof(CVoxelOccupancy);

		page.dEmptySegmentPos = reinterpret_cast<unsigned char*>(dPtr + offset);
		offset += GIVoxelPages::SegmentSize * sizeof(unsigned char);

		page.dSegmentInfo = reinterpret_cast<CSegmentInfo*>(dPtr + offset);
		offset += GIVoxelPages::SegmentSize * sizeof(CSegmentInfo);

		page.dEmptySegmentStackSize = GIVoxelPages::SegmentPerPage;
		pages.push_back(page);
	}
	assert(offset == pageData.Size());

	// KC to Initialize Empty Segment Stack
	int gridSize = CudaInit::GenBlockSizeSmall(static_cast<uint32_t>(pageCount * GIVoxelPages::SegmentPerPage));
	int blockSize = CudaInit::TBP;
	InitializePage<<<gridSize, blockSize>>>(pages.front().dEmptySegmentPos, pageCount);
	CUDA_KERNEL_CHECK();
}

GIVoxelPages::MultiPage::MultiPage(MultiPage&& other)
	: pageData(std::move(other.pageData))
	, pages(std::move(other.pages))
{}

size_t GIVoxelPages::MultiPage::PageCount() const
{
	return pages.size();
}

const std::vector<CVoxelPage>& GIVoxelPages::MultiPage::Pages() const
{
	return pages;
}

uint16_t GIVoxelPages::PackSegmentInfo(const uint8_t cascadeId,
									   const CObjectType type,
									   const CSegmentOccupation occupation,
									   const bool firstOccurance)
{
	// MSB to LSB 
	// 2 bit cascadeId
	// 2 bit object type 
	// 2 bit segment occupation
	uint16_t packed = 0;
	packed |= (static_cast<uint16_t>(cascadeId) & 0x0003) << 14;
	packed |= (static_cast<uint16_t>(type) & 0x0003) << 12;
	packed |= (static_cast<uint16_t>(occupation) & 0x0003) << 10;
	packed |= (static_cast<uint16_t>(firstOccurance) & 0x0001) << 9;
	return packed;
}

void GIVoxelPages::GenerateGPUData(const GIVoxelCache& cache)
{
	// Generate SegInfos
	std::vector<CVoxelGrid> grids;
	std::vector<CSegmentInfo> segInfos;
	std::vector<std::vector<bool>> checkBase(batches->size());

	for(uint32_t cascadeId = 0; cascadeId < svoParams->CascadeCount; cascadeId++)
	{
		CVoxelGrid grid = {};
		grid.depth = svoParams->CascadeBaseLevel + svoParams->CascadeCount - cascadeId - 1;
		grid.dimension = 
		{
			svoParams->CascadeBaseLevelSize,
			svoParams->CascadeBaseLevelSize,
			svoParams->CascadeBaseLevelSize
		};
		grid.position = {0.0f, 0.0f, 0.0f};
		grid.span = svoParams->BaseSpan * static_cast<float>(1 << cascadeId);
		grids.push_back(grid);
		
		for(uint32_t batchId = 0; batchId < batches->size(); batchId++)
		{
			if((*batches)[batchId]->DrawCount() == 0) continue;
			if(cascadeId == 0) checkBase[batchId].resize((*batches)[batchId]->DrawCount(), true);
			
			bool nonRigid = (*batches)[batchId]->MeshType() == MeshBatchType::SKELETAL;
			const std::vector<CMeshVoxelInfo> voxInfo = cache.CopyMeshObjectInfo(cascadeId, batchId);

			for(uint32_t objId = 0; objId < voxInfo.size(); objId++)
			{
				const CMeshVoxelInfo& info = voxInfo[objId];
				bool firstOccurance = false;
				if(info.voxCount != 0 && checkBase[batchId][objId] == true)
				{
					checkBase[batchId][objId] = false;
					firstOccurance = true;
				}

				uint32_t segmentCount = (info.voxCount + SegmentSize - 1) / SegmentSize;
				for(uint32_t segId = 0; segId < segmentCount; segId++)
				{
					CObjectType objType = (nonRigid) ? CObjectType::SKEL_DYNAMIC : CObjectType::DYNAMIC;

					CSegmentInfo segInfo;
					segInfo.batchId = static_cast<uint16_t>(batchId);
					segInfo.objectSegmentId = static_cast<uint16_t>(segId);
					segInfo.objId = static_cast<uint16_t>(objId);
					segInfo.packed = PackSegmentInfo(static_cast<uint8_t>(cascadeId), objType,
													 CSegmentOccupation::OCCUPIED, 
													 firstOccurance);

					segInfos.push_back(segInfo);
				}
			}
		}
	}

	// Determine Buffer Size
	size_t bufferSize = segInfos.size() * (sizeof(CSegmentInfo) +
										   sizeof(ushort2));
	bufferSize += batches->size() * sizeof(BatchOGLData);
	bufferSize += svoParams->CascadeCount * sizeof(CVoxelGrid);

	// Send Data to Buffer
	gpuData.Resize(bufferSize);
	size_t bufferOffset = 0;
	// Grids
	CUDA_CHECK(cudaMemcpy(gpuData.Data() + bufferOffset,
						  reinterpret_cast<void*>(grids.data()),
						  grids.size() * sizeof(CVoxelGrid),
						  cudaMemcpyHostToDevice));
	dVoxelGrids = reinterpret_cast<CVoxelGrid*>(gpuData.Data() + bufferOffset);
	bufferOffset += grids.size() * sizeof(CVoxelGrid);
	// OGL Data
	CUDA_CHECK(cudaMemset(gpuData.Data() + bufferOffset, 0,
						  batches->size() * sizeof(BatchOGLData)));
	dBatchOGLData = reinterpret_cast<BatchOGLData*>(gpuData.Data() + bufferOffset);
	bufferOffset += batches->size() * sizeof(BatchOGLData);
	// Segments Alloc
	CUDA_CHECK(cudaMemset(gpuData.Data() + bufferOffset, 0xFFFFFFFF,
						  segInfos.size() * sizeof(ushort2)));
	dSegmentAllocInfo = reinterpret_cast<ushort2*>(gpuData.Data() + bufferOffset);
	bufferOffset += segInfos.size() * sizeof(ushort2);
	// Segments
	CUDA_CHECK(cudaMemcpy(gpuData.Data() + bufferOffset,
						  reinterpret_cast<void*>(segInfos.data()),
						  segInfos.size() * sizeof(CSegmentInfo),
						  cudaMemcpyHostToDevice));
	dSegmentInfo = reinterpret_cast<CSegmentInfo*>(gpuData.Data() + bufferOffset);
	bufferOffset += segInfos.size() * sizeof(CSegmentInfo);
	assert(bufferOffset == gpuData.Size());
	segmentAmount = static_cast<uint32_t>(segInfos.size());
}

void GIVoxelPages::AllocatePages(size_t voxelCapacity)
{
	size_t pageCount = (voxelCapacity + PageSize - 1) / PageSize;
	size_t oldSize = dPages.Size();

	hPages.emplace_back(pageCount);
	dPages.Resize(oldSize + hPages.back().PageCount());
	dPages.Assign(oldSize, hPages.back().PageCount(), hPages.back().Pages().data());
}

void GIVoxelPages::MapOGLResources()
{
	CUDA_CHECK(cudaGraphicsMapResources(static_cast<int>(batchOGLResources.size()), batchOGLResources.data()));

	std::vector<BatchOGLData> newOGLData;
	size_t batchIndex = 0;
	for(size_t i = 0; i < batches->size(); i++)
	{
		MeshBatchI& currentBatch = *(*batches)[i];
		if(currentBatch.DrawCount() == 0)
		{
			newOGLData.push_back({});
			continue;
		}

		size_t size;
		uint8_t* glPointer = nullptr;
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&glPointer),
														&size, batchOGLResources[batchIndex]));

		size_t aabbByteOffset = (*batches)[i]->getDrawBuffer().getAABBOffset();
		size_t modelTransformByteOffset = (*batches)[i]->getDrawBuffer().getModelTransformOffset();
		size_t modelTransformIndexByteOffset = (*batches)[i]->getDrawBuffer().getModelTransformIndexOffset();

		BatchOGLData batchGL = {};
		batchGL.dAABBs = reinterpret_cast<CAABB*>(glPointer + aabbByteOffset);
		batchGL.dModelTransforms = reinterpret_cast<CModelTransform*>(glPointer + modelTransformByteOffset);
		batchGL.dModelTransformIndices = reinterpret_cast<uint32_t*>(glPointer + modelTransformIndexByteOffset);
		
		batchIndex++;
		if((*batches)[i]->MeshType() == MeshBatchType::SKELETAL)
		{
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&glPointer),
															&size, batchOGLResources[batchIndex]));
			batchGL.dJointTransforms = reinterpret_cast<CJointTransform*>(glPointer);
			batchIndex++;
		}
		newOGLData.push_back(batchGL);
	}

	// Copy generated pointers to GPU
	CUDA_CHECK(cudaMemcpy(dBatchOGLData,
						  newOGLData.data(),
						  batches->size() * sizeof(BatchOGLData),
						  cudaMemcpyHostToDevice));
}

void GIVoxelPages::UnmapOGLResources()
{
	CUDA_CHECK(cudaGraphicsUnmapResources(static_cast<int>(batchOGLResources.size()), batchOGLResources.data()));
}

double GIVoxelPages::VoxelizeFast()
{
	return 0.0f;
}

void GIVoxelPages::Update(double& ioTime,
											   double& transTime,
											   const GIVoxelCache& caches,
											   const IEVector3& camPos,
											   bool doTiming,
											   bool useCache)
{
	UpdateGridPositions(camPos);
	if(useCache)
	{
		MapOGLResources();
		ioTime = VoxelIO(doTiming);
		transTime = Transform(caches, doTiming);
		UnmapOGLResources();
	}
	else
	{
		transTime = VoxelizeFast();
	}
}

GIVoxelPages::GIVoxelPages()
	: batches(nullptr)
	, svoParams(nullptr)
	, segmentAmount(0)
	, dVoxelGrids(nullptr)
	, dBatchOGLData(nullptr)
	, dSegmentInfo(nullptr)
	, dSegmentAllocInfo(nullptr)

{}

GIVoxelPages::GIVoxelPages(const GIVoxelCache& cache, 
						   const std::vector<MeshBatchI*>* batches,
						   const OctreeParameters& octreeParams)
	: batches(batches)
	, svoParams(&octreeParams)
	, segmentAmount(0)
	, dVoxelGrids(nullptr)
	, dBatchOGLData(nullptr)
	, dSegmentInfo(nullptr)
	, dSegmentAllocInfo(nullptr)
{
	for(uint32_t i = 0; i < batches->size(); i++)
	{	
		MeshBatchI& batch = *(*batches)[i];
		if(batch.DrawCount() == 0) continue;

		GLuint bufferId = batch.getDrawBuffer().getGLBuffer();
		cudaGraphicsResource_t glResource;
		CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&glResource, 
												bufferId,
												cudaGraphicsMapFlagsReadOnly));
		batchOGLResources.push_back(glResource);

		if(batch.MeshType() == MeshBatchType::SKELETAL)
		{
			GLuint jointBuffer = static_cast<MeshBatchSkeletal&>(batch).getJointTransforms().getGLBuffer();
			CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&glResource,
													jointBuffer,
													cudaGraphicsMapFlagsReadOnly));
			batchOGLResources.push_back(glResource);
		}
	}
	GenerateGPUData(cache);
	AllocatePages(segmentAmount * SegmentSize);
}

GIVoxelPages::GIVoxelPages(GIVoxelPages&& other)
	: batches(other.batches)
	, svoParams(other.svoParams)
	, segmentAmount(other.segmentAmount)
	, outermostGridPosition(other.outermostGridPosition)
	, gpuData(std::move(other.gpuData))
	, dVoxelGrids(other.dVoxelGrids)
	, dBatchOGLData(other.dBatchOGLData)
	, dSegmentInfo(other.dSegmentInfo)
	, dSegmentAllocInfo(other.dSegmentAllocInfo)
	, hPages(std::move(other.hPages))
	, dPages(std::move(other.dPages))
	, batchOGLResources(std::move(other.batchOGLResources))
	, pageRenderer(std::move(other.pageRenderer))
{
	assert(other.batchOGLResources.empty());
}

GIVoxelPages& GIVoxelPages::operator=(GIVoxelPages&& other)
{
	assert(&other != this);
	for(cudaGraphicsResource_t resc : batchOGLResources)
	{
		CUDA_CHECK(cudaGraphicsUnregisterResource(resc));
	}

	batches = other.batches;
	svoParams = other.svoParams;
	segmentAmount = other.segmentAmount;
	outermostGridPosition = other.outermostGridPosition;
	gpuData = std::move(other.gpuData);
	dVoxelGrids = other.dVoxelGrids;
	dBatchOGLData = other.dBatchOGLData;
	dSegmentInfo = other.dSegmentInfo;
	dSegmentAllocInfo = other.dSegmentAllocInfo;
	hPages = std::move(other.hPages);
	dPages = std::move(other.dPages);
	batchOGLResources = std::move(other.batchOGLResources);
	pageRenderer = std::move(other.pageRenderer);
	return *this;
}

GIVoxelPages::~GIVoxelPages()
{
	for(cudaGraphicsResource_t resc : batchOGLResources)
	{
		CUDA_CHECK(cudaGraphicsUnregisterResource(resc));
	}
}

void GIVoxelPages::UpdateGridPositions(const IEVector3& cameraPos)
{
	// Calculate outermost span position
	float outerSpan = svoParams->BaseSpan * static_cast<float>(1 << (svoParams->CascadeCount - 1));
	IEVector3 voxelCornerPos = cameraPos - outerSpan * (svoParams->CascadeBaseLevelSize - 1) * 0.5f;

	// Align outermost cascade
	// TODO: Better solution for higher level voxel jittering
    float rootSnapLevelMultiplier = static_cast<float>(0x1 << 3);

	// Removes Jitterin on base cascade level
	float snapSpan = outerSpan * rootSnapLevelMultiplier;
	voxelCornerPos[0] -= std::fmod(voxelCornerPos[0] + snapSpan * 0.5f, snapSpan);
	voxelCornerPos[1] -= std::fmod(voxelCornerPos[1] + snapSpan * 0.5f, snapSpan);
	voxelCornerPos[2] -= std::fmod(voxelCornerPos[2] + snapSpan * 0.5f, snapSpan);

	//// Grid Aligned Center
	//IEVector3 voxelCenter = voxelCornerPos + outerSpan * (svoParams->CascadeBaseLevelSize - 1) * 0.5f;
	//std::vector<IEVector3> positions(svoParams->CascadeCount);
	//for(uint32_t i = 0; i < svoParams->CascadeCount; i++)
	//{
	//	float multiplier = (0x1 << i) * (svoParams->CascadeBaseLevelSize - 1) * 0.5f;
	//	positions[i] = voxelCenter - multiplier;
	//}

	// Now align inner cascades according to outermost
	// In all system cacades and its data lied from inner to outer
	float baseHalf = svoParams->BaseSpan * 0.5f * svoParams->CascadeBaseLevelSize;
	float seriesTotal = IEMathFunctions::GeomSeries(svoParams->CascadeCount - 2, 2.0f);
	std::vector<IEVector3> positions(svoParams->CascadeCount);
	for(uint32_t i = 0; i < svoParams->CascadeCount; i++)
	{
		int32_t termLast = i - 1;
		float lastTermSum = (termLast >= 0) ?  IEMathFunctions::GeomSeries(termLast, 2.0f) : 0;
		float subSeries = seriesTotal - lastTermSum;
		float displacement = subSeries * baseHalf;
		positions[i] = voxelCornerPos + displacement;
	}

	outermostGridPosition = positions.back();

	// Copy new positions
	CUDA_CHECK(cudaMemcpy2D(dVoxelGrids, sizeof(CVoxelGrid),
							positions.data(), sizeof(IEVector3),
							sizeof(IEVector3), svoParams->CascadeCount,
							cudaMemcpyHostToDevice));
}

double GIVoxelPages::VoxelIO(bool doTiming)
{
	CudaTimer t;
	if(doTiming) t.Start();
	
	// KC
	int gridSize = CudaInit::GenBlockSizeSmall(static_cast<int>(segmentAmount));
	int blockSize = CudaInit::TBPSmall;
	// Voxel I-O (Deallocate first then allocate)
	VoxelDeallocate<<<gridSize, blockSize>>>(// Voxel System
										     dPages.Data(),
										     dVoxelGrids,
										     // Helper Structures		
										     dSegmentAllocInfo,
										     dSegmentInfo,
										     // Per Object Related
										     dBatchOGLData,
										     // Limits
										     segmentAmount);

	VoxelAllocate<<<gridSize, blockSize>>>(// Voxel System
										   dPages.Data(),
										   dVoxelGrids,
										   // Helper Structures		
										   dSegmentAllocInfo,
										   dSegmentInfo,
										   // Per Object Related
										   dBatchOGLData,
										   // Limits
										   segmentAmount,
										   static_cast<uint32_t>(dPages.Size()));
	CUDA_KERNEL_CHECK();
	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMilliS();
	}
	return 0.0;
}

double GIVoxelPages::Transform(const GIVoxelCache& cache,
							   bool doTiming)
{
	CudaTimer t;
	if(doTiming) t.Start();

	// KC
	int gridSize = CudaInit::GenBlockSizeSmall(static_cast<int>(dPages.Size() * PageSize));
	int blockSize = CudaInit::TBPSmall;
	VoxelTransform<<<gridSize, blockSize>>>(// Voxel Pages
										    dPages.Data(),
										    dVoxelGrids,
										    // OGL Related
										    dBatchOGLData,
										    // Voxel Cache Related
										    cache.getDeviceCascadePointersDevice().Data(),
										    // Limits
										    static_cast<uint32_t>(batches->size()));
	cudaDeviceSynchronize();
	CUDA_KERNEL_CHECK();
	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMilliS();
	}
	return 0.0;
}

uint64_t GIVoxelPages::MemoryUsage() const
{
	size_t totalSize = gpuData.Size();
	totalSize += dPages.Size() * sizeof(CVoxelPage);
	totalSize += dPages.Size() * PageSize * (sizeof(CVoxelPos) +
											 sizeof(CVoxelNorm) +
											 sizeof(CVoxelOccupancy));
	totalSize += dPages.Size() * SegmentPerPage * (sizeof(unsigned char) +
												   sizeof(CSegmentInfo));
	return totalSize;
}

uint32_t GIVoxelPages::PageCount() const
{
	return static_cast<uint32_t>(dPages.Size());
}

void GIVoxelPages::DumpPageSegments(const char* fileName, size_t offset, size_t pageCount) const
{
	if(pageCount == 0) pageCount = dPages.Size() - offset;
	assert(offset + pageCount <= dPages.Size());

	std::vector<CVoxelPage> pages(pageCount);
	CUDA_CHECK(cudaMemcpy(pages.data(), dPages.Data() + offset, 
						  pageCount * sizeof(CVoxelPage),
						  cudaMemcpyDeviceToHost));

	std::vector<CSegmentInfo> infos(pageCount * SegmentPerPage);
	for(size_t i = 0; i < pageCount; i++)
	{
		const CVoxelPage& p = pages[i];
		CUDA_CHECK(cudaMemcpy(infos.data() + i * SegmentPerPage,
							  p.dSegmentInfo,
							  SegmentPerPage * sizeof(CSegmentInfo),
							  cudaMemcpyDeviceToHost));
	}


	std::ofstream fOut;
	fOut.open(fileName);
	for(const CSegmentInfo& data : infos)
	{
		fOut << std::uppercase << std::hex << data;
		fOut << "\t\t\t" << std::nouppercase << std::dec << data;
		fOut << std::endl;
	}
}

void GIVoxelPages::DumpPageEmptyPositions(const char* fileName, size_t offset, size_t pageCount) const
{
	if(pageCount == 0) pageCount = dPages.Size() - offset;
	assert(offset + pageCount <= dPages.Size());

	std::vector<CVoxelPage> pages(pageCount);
	CUDA_CHECK(cudaMemcpy(pages.data(), dPages.Data() + offset,
						  pageCount * sizeof(CVoxelPage),
						  cudaMemcpyDeviceToHost));

	std::vector<unsigned char> emptySpots(pageCount * SegmentPerPage);
	for(size_t i = 0; i < pageCount; i++)
	{
		const CVoxelPage& p = pages[i];
		CUDA_CHECK(cudaMemcpy(emptySpots.data() + i * SegmentPerPage,
							  p.dEmptySegmentPos,
							  SegmentPerPage * sizeof(unsigned char),
							  cudaMemcpyDeviceToHost));
	}

	std::ofstream fOut;
	fOut.open(fileName);
	for(const unsigned char& data : emptySpots)
	{
		fOut << std::uppercase << std::hex << static_cast<int>(data);
		fOut << "\t\t\t" << std::nouppercase << std::dec << static_cast<int>(data);
		fOut << std::endl;
	}
}

void GIVoxelPages::DumpSegmentAllocation(const char* fileName, size_t offset, size_t segmentCount) const
{
	if(segmentCount == 0) segmentCount = segmentAmount - offset;
	assert(offset + segmentCount <= segmentAmount);

	std::vector<ushort2> segments(segmentCount);
	CUDA_CHECK(cudaMemcpy(segments.data(), dSegmentInfo + offset,
						  segmentCount * sizeof(ushort2),
						  cudaMemcpyDeviceToHost));

	std::ofstream fOut;
	fOut.open(fileName);
	for(const ushort2& data : segments)
	{
		fOut << std::uppercase << std::hex << data;
		fOut << "\t\t\t" << std::nouppercase << std::dec << data;
		fOut << std::endl;
	}
}

void GIVoxelPages::DumpSegmentInfo(const char* fileName, size_t offset, size_t segmentCount) const
{
	if(segmentCount == 0) segmentCount = segmentAmount - offset;
	assert(offset + segmentCount <= segmentAmount);

	std::vector<CSegmentInfo> segments(segmentCount);
	CUDA_CHECK(cudaMemcpy(segments.data(), dSegmentInfo + offset,
						  segmentCount * sizeof(CSegmentInfo),
						  cudaMemcpyDeviceToHost));

	std::ofstream fOut;
	fOut.open(fileName);
	for(const CSegmentInfo& data : segments)
	{
		fOut << std::uppercase << std::hex << data;
		fOut << "\t\t\t" << std::nouppercase << std::dec << data;
		fOut << std::endl;
	}
}

void GIVoxelPages::AllocateDraw()
{
	if(!pageRenderer.Allocated())
	{
		pageRenderer = PageRenderer(*this);
	}
}

double GIVoxelPages::Draw(bool doTiming,
						  uint32_t cascadeCount,
						  VoxelRenderType renderType,
						  const Camera& camera,
						  const GIVoxelCache& cache)
{
	return pageRenderer.Draw(doTiming, cascadeCount, renderType, camera, cache, *this);
}

void GIVoxelPages::DeallocateDraw()
{
	if(pageRenderer.Allocated())
	{
		pageRenderer = PageRenderer();
	}
}

const CVoxelPageConst* GIVoxelPages::getVoxelPagesDevice() const
{
	return reinterpret_cast<const CVoxelPageConst*>(dPages.Data());
}

const CVoxelGrid* GIVoxelPages::getVoxelGridsDevice() const
{
	return dVoxelGrids;
}

const IEVector3& GIVoxelPages::getOutermostGridPosition() const
{
	return outermostGridPosition;
}