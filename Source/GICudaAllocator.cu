#include "GICudaAllocator.h"
#include "GICudaStructMatching.h"
#include <cuda_gl_interop.h>

GICudaAllocator::GICudaAllocator()
	: totalObjectCount(0)
{
	cudaGLSetGLDevice(0);
}

void GICudaAllocator::LinkOGLVoxelCache(GLuint batchAABBBuffer,
											GLuint batchTransformBuffer,
											GLuint relativeTransformBuffer,
											GLuint infoBuffer,
											GLuint voxelBuffer,
											GLuint voxelRenderBuffer,
											size_t objCount)
{
	rTransformLinks.emplace_back(nullptr);
	transformLinks.emplace_back(nullptr);
	aabbLinks.emplace_back(nullptr);
	objectInfoLinks.emplace_back(nullptr);
	cacheLinks.emplace_back(nullptr);
	cacheRenderLinks.emplace_back(nullptr);

	cudaGraphicsGLRegisterBuffer(&rTransformLinks.back(), relativeTransformBuffer, cudaGraphicsMapFlagsReadOnly);
	cudaGraphicsGLRegisterBuffer(&transformLinks.back(), batchTransformBuffer, cudaGraphicsMapFlagsReadOnly);
	cudaGraphicsGLRegisterBuffer(&aabbLinks.back(), batchAABBBuffer, cudaGraphicsMapFlagsReadOnly);
	cudaGraphicsGLRegisterBuffer(&objectInfoLinks.back(), infoBuffer, cudaGraphicsMapFlagsReadOnly);

	cudaGraphicsGLRegisterBuffer(&cacheLinks.back(), voxelBuffer, cudaGraphicsMapFlagsReadOnly);
	cudaGraphicsGLRegisterBuffer(&cacheRenderLinks.back(), voxelRenderBuffer, cudaGraphicsMapFlagsReadOnly);

	objectCounts.emplace_back(objCount);
	totalObjectCount += objCount;
}

void GICudaAllocator::SetupPointersDevicePointers()
{
	cudaGraphicsMapResources(rTransformLinks.size(), rTransformLinks.data());
	cudaGraphicsMapResources(transformLinks.size(), transformLinks.data());
	cudaGraphicsMapResources(aabbLinks.size(), aabbLinks.data());
	cudaGraphicsMapResources(objectInfoLinks.size(), objectInfoLinks.data());

	cudaGraphicsMapResources(cacheLinks.size(), cacheLinks.data());
	cudaGraphicsMapResources(cacheRenderLinks.size(), cacheRenderLinks.data());

	thrust::host_vector<CObjectTransform*> hRelativeTransforms;
	thrust::host_vector<CObjectTransform*> hTransforms;
	thrust::host_vector<CObjectAABB*> hObjectAABB;
	thrust::host_vector<CObjectVoxelInfo*> hObjectInfo;
			
	thrust::host_vector<CVoxelPacked*> hObjCache;
	thrust::host_vector<CVoxelRender*> hObjRenderCache;

	size_t size = 0;
	for(unsigned int i = 0; i < objectCounts.size(); i++)
	{
		hRelativeTransforms.push_back(nullptr);
		hTransforms.push_back(nullptr);
		hObjectAABB.push_back(nullptr);
		hObjectInfo.push_back(nullptr);

		hObjCache.push_back(nullptr);
		hObjRenderCache.push_back(nullptr);

		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hRelativeTransforms.back()), &size, rTransformLinks[i]);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hTransforms.back()), &size, transformLinks[i]);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjectAABB.back()), &size, aabbLinks[i]);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjectInfo.back()), &size, objectInfoLinks[i]);

		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjCache.back()), &size, cacheLinks[i]);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjRenderCache.back()), &size, cacheRenderLinks[i]);
	}

	// Dat Sent to GPU
	dRelativeTransforms = hRelativeTransforms;
	dTransforms = hTransforms;
	dObjectAABB = hObjectAABB;
	dObjectInfo = hObjectInfo;

	dObjCache = hObjCache;
	dObjRenderCache = hObjRenderCache;
}

void GICudaAllocator::ClearDevicePointers()
{
	dRelativeTransforms.clear();
	dTransforms.clear();
	dObjectAABB.clear();
	dObjectInfo.clear();

	dObjCache.clear();
	dObjRenderCache.clear();

	cudaGraphicsUnmapResources(rTransformLinks.size(), rTransformLinks.data());
	cudaGraphicsUnmapResources(transformLinks.size(), transformLinks.data());
	cudaGraphicsUnmapResources(aabbLinks.size(), aabbLinks.data());
	cudaGraphicsUnmapResources(objectInfoLinks.size(), objectInfoLinks.data());

	cudaGraphicsUnmapResources(cacheLinks.size(), cacheLinks.data());
	cudaGraphicsUnmapResources(cacheRenderLinks.size(), cacheRenderLinks.data());
}


void GICudaAllocator::AddVoxelPage(size_t count)
{

	for(unsigned int i = 0; i < count; i++)
	{
		// Allocating Page
		hPageData.emplace_back(thrust::device_vector<CVoxelPacked>({0}, GI_PAGE_SIZE),
							   thrust::device_vector<CVoxelRender>({0}, GI_PAGE_SIZE),
							   thrust::device_vector<unsigned int>({0}, GI_PAGE_SIZE),
							   thrust::device_vector<unsigned int>(0, GI_BLOCK_PER_PAGE));

		CVoxelPage voxData =
		{
			thrust::raw_pointer_cast(hPageData.back().dVoxelPage.data()),
			thrust::raw_pointer_cast(hPageData.back().dVoxelPageRender.data()),
			thrust::raw_pointer_cast(hPageData.back().dVoxelState.data()),
			thrust::raw_pointer_cast(hPageData.back().dEmptySegmentPos.data()),
			0
		};
		hVoxelPages.push_back(voxData);
		dVoxelPages.push_back(voxData);
	}
}

const CObjectTransform** GICudaAllocator::GetRelativeTransformsDevice() 
{
	return thrust::raw_pointer_cast(dRelativeTransforms.data());
}

const CObjectTransform** GICudaAllocator::GetTransformsDevice()
{
	return thrust::raw_pointer_cast(dTransforms.data());
}

const CObjectAABB** GICudaAllocator::GetObjectAABBDevice()
{
	return thrust::raw_pointer_cast(dObjectAABB.data());
}

const CObjectVoxelInfo** GICudaAllocator::GetObjectInfoDevice()
{
	return thrust::raw_pointer_cast(dObjectInfo.data());
}

const CVoxelPacked** GICudaAllocator::GetObjCacheDevice()
{
	return thrust::raw_pointer_cast(dObjCache.data());
}

const CVoxelRender** GICudaAllocator::GetObjRenderCacheDevice()
{
	return thrust::raw_pointer_cast(dObjRenderCache.data());
}

CVoxelPage* GICudaAllocator::GetVoxelPagesDevice()
{
	return thrust::raw_pointer_cast(dVoxelPages.data());
}
