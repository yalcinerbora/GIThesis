#include "GISparseVoxelOctree.h"

//void GICudaAllocator::LinkSceneShadowMapArray(GLuint shadowMapArray)
//{
//	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&sceneShadowMapLink,
//	//									   shadowMapArray,
//	//									   GL_TEXTURE_2D_ARRAY,
//	//									   cudaGraphicsRegisterFlagsReadOnly));
//}
//
//void GICudaAllocator::LinkSceneGBuffers(GLuint depthTex,
//										GLuint normalTex,
//										GLuint lightIntensityTex)
//{
//	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&depthBuffLink,
//	//										depthTex,
//	//										GL_TEXTURE_2D,
//	//										cudaGraphicsRegisterFlagsReadOnly));
//	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&normalBuffLink,
//	//										normalTex,
//	//										GL_TEXTURE_2D,
//	//										cudaGraphicsRegisterFlagsReadOnly));
//	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&lightIntensityLink,
//	//										lightIntensityTex,
//	//										GL_TEXTURE_2D,
//	//										cudaGraphicsRegisterFlagsSurfaceLoadStore));
//}
//
//void GICudaAllocator::UnLinkGBuffers()
//{
//	//CUDA_CHECK(cudaGraphicsUnregisterResource(depthBuffLink));
//	//CUDA_CHECK(cudaGraphicsUnregisterResource(normalBuffLink));
//	//CUDA_CHECK(cudaGraphicsUnregisterResource(lightIntensityLink));
//}

// Textures
//cudaArray_t texArray;
//cudaMipmappedArray_t mipArray;
//cudaResourceDesc resDesc = {};
//cudaTextureDesc texDesc = {};

//resDesc.resType = cudaResourceTypeMipmappedArray;

//texDesc.addressMode[0] = cudaAddressModeWrap;
//texDesc.addressMode[1] = cudaAddressModeWrap;
//texDesc.filterMode = cudaFilterModePoint;
//texDesc.readMode = cudaReadModeElementType;
//texDesc.normalizedCoords = 1;

//CUDA_CHECK(cudaGraphicsMapResources(1, &sceneShadowMapLink));
//CUDA_CHECK(cudaGraphicsResourceGetMappedMipmappedArray(&mipArray, sceneShadowMapLink));
//resDesc.res.mipmap.mipmap = mipArray;
//CUDA_CHECK(cudaCreateTextureObject(&shadowMaps, &resDesc, &texDesc, nullptr));

//texDesc.normalizedCoords = 1;
//resDesc.resType = cudaResourceTypeArray;

//CUDA_CHECK(cudaGraphicsMapResources(1, &depthBuffLink));
//CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texArray, depthBuffLink, 0, 0));
//resDesc.res.array.array = texArray;
//CUDA_CHECK(cudaCreateTextureObject(&depthBuffer, &resDesc, &texDesc, nullptr));

//CUDA_CHECK(cudaGraphicsMapResources(1, &normalBuffLink));
//CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texArray, normalBuffLink, 0, 0));
//resDesc.res.array.array = texArray;
//CUDA_CHECK(cudaCreateTextureObject(&normalBuffer, &resDesc, &texDesc, nullptr));

//CUDA_CHECK(cudaGraphicsMapResources(1, &lightIntensityLink));
//CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texArray, lightIntensityLink, 0, 0));
//resDesc.res.array.array = texArray;
//CUDA_CHECK(cudaCreateSurfaceObject(&lightIntensityBuffer, &resDesc));