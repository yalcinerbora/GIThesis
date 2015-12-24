//#include "SVOTexPool.h"
//
//GLuint tex;
//glGenTextures(1, &tex);
//
//glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
//glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_SPARSE_ARB, GL_TRUE);
//
//glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_UNSIGNED_INT, 1000, 1000, 256);
//
//glTexPageCommitmentARB(GL_TEXTURE_2D_ARRAY,
//					   1, 0, 0,
//					   128, 1000, 1000, 1, true);
//
//
//cudaGraphicsResource_t gr;
//CUDA_CHECK(cudaGraphicsGLRegisterImage(&gr, tex, GL_TEXTURE_2D_ARRAY, cudaGraphicsMapFlagsWriteDiscard));
