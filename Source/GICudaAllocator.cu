#include "GICudaAllocator.h"
#include <cuda_gl_interop.h>

GICudaAllocator::GICudaAllocator()
{
	cudaGLSetGLDevice(0);
}

void GICudaAllocator::LinkOGLVoxelCache(GLuint batchAABBBuffer,
										GLuint batchTransformBuffer,
										GLuint relativeTransformBuffer,
										GLuint infoBuffer,
										GLuint voxelBuffer,
										GLuint voxelRenderBuffer)
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
}

void GICudaAllocator::SetupPointersDevicePointers()
{

}

void GICudaAllocator::ClearDevicePointers()
{

}


void GICudaAllocator::AddVoxelPage(size_t count)
{

}

void GICudaAllocator::ShrinkVoxelPages(size_t pageCount)
{

}