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

void GICudaAllocator::LinkSceneShadowMapArray(const std::vector<GLuint>& shadowMaps)
{
	cudaGraphicsResource* resource = nullptr;
	for(unsigned int i = 0; i < shadowMaps.size(); i++)
	{
		cudaGraphicsGLRegisterImage(&resource,
									shadowMaps[i],
									GL_TEXTURE_CUBE_MAP,
									cudaGraphicsRegisterFlagsReadOnly);
		sceneShadowMapLinks.push_back(resource);
	}
}

void GICudaAllocator::LinkSceneGBuffers(GLuint depthTex,
										GLuint normalTex,
										GLuint lightIntensityTex)
{
	cudaGraphicsGLRegisterImage(&depthBuffLink,
								depthTex,
								GL_TEXTURE_2D,
								cudaGraphicsRegisterFlagsReadOnly);
	cudaGraphicsGLRegisterImage(&normalBuffLink,
								normalTex,
								GL_TEXTURE_2D,
								cudaGraphicsRegisterFlagsReadOnly);
	cudaGraphicsGLRegisterImage(&lightIntensityLink,
								lightIntensityTex,
								GL_TEXTURE_2D,
								cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

void GICudaAllocator::SetupPointersDevicePointers()
{
	cudaGraphicsMapResources(static_cast<int>(rTransformLinks.size()), rTransformLinks.data());
	cudaGraphicsMapResources(static_cast<int>(transformLinks.size()), transformLinks.data());
	cudaGraphicsMapResources(static_cast<int>(aabbLinks.size()), aabbLinks.data());
	cudaGraphicsMapResources(static_cast<int>(objectInfoLinks.size()), objectInfoLinks.data());

	cudaGraphicsMapResources(static_cast<int>(cacheLinks.size()), cacheLinks.data());
	cudaGraphicsMapResources(static_cast<int>(cacheRenderLinks.size()), cacheRenderLinks.data());

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

	//// Data Sent to GPU
	dRelativeTransforms = hRelativeTransforms;
	dTransforms = hTransforms;
	dObjectAABB = hObjectAABB;
	dObjectInfo = hObjectInfo;

	dObjCache = hObjCache;
	dObjRenderCache = hObjRenderCache;


	// Textures
	cudaArray* texArray = nullptr;
	cudaResourceDesc resDesc = {};
	cudaTextureDesc texDesc = {};

	resDesc.res.array.array = texArray;
	resDesc.resType = cudaResourceTypeArray;

	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 0;

	cudaGraphicsMapResources(static_cast<int>(sceneShadowMapLinks.size()), sceneShadowMapLinks.data());
	for(unsigned int i = 0; i < sceneShadowMapLinks.size(); i++)
	{
		cudaGraphicsSubResourceGetMappedArray(&texArray, sceneShadowMapLinks[i], 0, 0);

		shadowMaps.emplace_back();
		cudaCreateTextureObject(&shadowMaps.back(), &resDesc, &texDesc, nullptr);
	}

	cudaGraphicsMapResources(1, &depthBuffLink);
	cudaGraphicsSubResourceGetMappedArray(&texArray, depthBuffLink, 0, 0);
	texDesc.readMode = cudaReadModeElementType;
	cudaCreateTextureObject(&depthBuffer, &resDesc, &texDesc, nullptr);

	cudaGraphicsMapResources(1, &normalBuffLink);
	cudaGraphicsSubResourceGetMappedArray(&texArray, normalBuffLink, 0, 0);
	texDesc.readMode = cudaReadModeElementType;
	cudaCreateTextureObject(&normalBuffer, &resDesc, &texDesc, nullptr);

	cudaGraphicsMapResources(1, &lightIntensityLink);
	cudaGraphicsSubResourceGetMappedArray(&texArray, lightIntensityLink, 0, 0);
	cudaCreateSurfaceObject(&lightIntensityBuffer, &resDesc);

}

void GICudaAllocator::ClearDevicePointers()
{
	dRelativeTransforms.clear();
	dTransforms.clear();
	dObjectAABB.clear();
	dObjectInfo.clear();

	dObjCache.clear();
	dObjRenderCache.clear();

	cudaDestroySurfaceObject(lightIntensityBuffer);
	cudaDestroyTextureObject(normalBuffer);
	cudaDestroyTextureObject(depthBuffer);

	for(unsigned int i = 0; i < shadowMaps.size(); i++)
	{
		cudaDestroyTextureObject(shadowMaps[i]);
	}
	shadowMaps.clear();

	cudaGraphicsUnmapResources(1, &depthBuffLink);
	cudaGraphicsUnmapResources(1, &normalBuffLink);
	cudaGraphicsUnmapResources(1, &lightIntensityLink);
	cudaGraphicsUnmapResources(static_cast<int>(sceneShadowMapLinks.size()), sceneShadowMapLinks.data());

	cudaGraphicsUnmapResources(static_cast<int>(rTransformLinks.size()), rTransformLinks.data());
	cudaGraphicsUnmapResources(static_cast<int>(transformLinks.size()), transformLinks.data());
	cudaGraphicsUnmapResources(static_cast<int>(aabbLinks.size()), aabbLinks.data());
	cudaGraphicsUnmapResources(static_cast<int>(objectInfoLinks.size()), objectInfoLinks.data());

	cudaGraphicsUnmapResources(static_cast<int>(cacheLinks.size()), cacheLinks.data());
	cudaGraphicsUnmapResources(static_cast<int>(cacheRenderLinks.size()), cacheRenderLinks.data());
}

void GICudaAllocator::AddVoxelPage(size_t count)
{
	for(unsigned int i = 0; i < count; i++)
	{
		// Allocating Page
		hPageData.emplace_back(CVoxelPageData
		{
			thrust::device_vector<CVoxelPacked>(GI_PAGE_SIZE),
			thrust::device_vector<unsigned int>(GI_BLOCK_PER_PAGE)
		});

		CVoxelPage voxData =
		{
			thrust::raw_pointer_cast(hPageData.back().dVoxelPage.data()),
			thrust::raw_pointer_cast(hPageData.back().dEmptySegmentList.data()),
			0
		};
		hVoxelPages.push_back(voxData);
	}
	dVoxelPages = hVoxelPages;
}

void GICudaAllocator::ResetSceneData()
{
	for(unsigned int i = 0; i < rTransformLinks.size(); i++)
	{
		cudaGraphicsUnregisterResource(rTransformLinks[i]);
		cudaGraphicsUnregisterResource(transformLinks[i]);
		cudaGraphicsUnregisterResource(aabbLinks[i]);
		cudaGraphicsUnregisterResource(objectInfoLinks[i]);

		cudaGraphicsUnregisterResource(cacheLinks[i]);
		cudaGraphicsUnregisterResource(cacheRenderLinks[i]);
	}

	for(unsigned int i = 0; i < sceneShadowMapLinks.size(); i++)
	{
		cudaGraphicsUnregisterResource(sceneShadowMapLinks[i]);
	}
	cudaGraphicsUnregisterResource(depthBuffLink);
	cudaGraphicsUnregisterResource(normalBuffLink);
	cudaGraphicsUnregisterResource(lightIntensityLink);

	rTransformLinks.clear();
	transformLinks.clear();
	aabbLinks.clear();
	objectInfoLinks.clear();

	cacheLinks.clear();
	cacheRenderLinks.clear();

	sceneShadowMapLinks.clear();
}

//const CObjectTransform** GICudaAllocator::GetRelativeTransformsDevice() 
//{
//	return thrust::raw_pointer_cast(dRelativeTransforms.data());
//}
//
//const CObjectTransform** GICudaAllocator::GetTransformsDevice()
//{
//	return thrust::raw_pointer_cast(dTransforms.data());
//}
//
//const CObjectAABB** GICudaAllocator::GetObjectAABBDevice()
//{
//	return thrust::raw_pointer_cast(dObjectAABB.data());
//}
//
//const CObjectVoxelInfo** GICudaAllocator::GetObjectInfoDevice()
//{
//	return thrust::raw_pointer_cast(dObjectInfo.data());
//}
//
//const CVoxelPacked** GICudaAllocator::GetObjCacheDevice()
//{
//	return thrust::raw_pointer_cast(dObjCache.data());
//}
//
//const CVoxelRender** GICudaAllocator::GetObjRenderCacheDevice()
//{
//	return thrust::raw_pointer_cast(dObjRenderCache.data());
//}

CVoxelPage* GICudaAllocator::GetVoxelPagesDevice()
{
	return thrust::raw_pointer_cast(dVoxelPages.data());
}
