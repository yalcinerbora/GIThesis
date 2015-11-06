/**



*/

#ifndef __GICUDASPARSEVOXELOCTREE_H__
#define __GICUDASPARSEVOXELOCTREE_H__

#include <cuda.h>
#include "CudaVector.cuh"
#include "SceneLights.h"
#include "VoxelDebugVAO.h"
#include "GICudaVoxelScene.h"
#include "CSVOTypes.cuh"

#define GI_DENSE_LEVEL 6
#define GI_DENSE_SIZE 64

class GICudaAllocator;
struct Camera;

class GISparseVoxelOctree
{
	private:
		std::vector<GICudaAllocator*>			allocators;			// Page Allocators
		std::vector<CVoxelGrid>					allocatorGrids;		// Allocator's Responsible Grids

		CSVOConstants							hSVOConstants;
		CudaVector<CSVOConstants>				dSVOConstants;

		// Debug Stuff
		StructuredBuffer<VoxelNormPos>			vaoNormPosData;
		StructuredBuffer<uchar4>				vaoColorData;

		// Texture copied from dSVO dense every frame
		cudaTextureObject_t						tSVODense;
		cudaArray_t								denseArray;

		// SVO Data
		CudaVector<CSVONode>					dSVO;				// Entire SVO
		CudaVector<CSVOColor>					dSVOColor;			// Entire SVO

		// SVO Ptrs
		CSVONode*								dSVODense;
		CSVONode*								dSVOSparse;
		CudaVector<unsigned int>				dSVOLevelStartIndices;
		CudaVector<unsigned int>				dSVONodeCountAtomic;

		// Inital Rays buffer
		GLuint									initalRayLink;
		cudaGraphicsResource_t					rayLinks;

		// Interop Data
		cudaGraphicsResource_t					shadowMapArrayTexLink;
		cudaGraphicsResource_t					lightBufferLink;
		cudaGraphicsResource_t					lightIntensityTexLink;
		
		void									ConstructDense();
		void									ConstructLevel(unsigned int levelIndex,
															   unsigned int allocatorIndex,
															   unsigned int cascadeNo);

	protected:

	public:
		// Constructors & Destructor
												GISparseVoxelOctree(GLuint lightIntensityTex);
												GISparseVoxelOctree(const GISparseVoxelOctree&) = delete;
		GISparseVoxelOctree&					operator=(const GISparseVoxelOctree&) = delete;
												~GISparseVoxelOctree();

		// Link Allocators and Adjust Size of the System
		void									 LinkAllocators(GICudaAllocator** newAllocators,
																size_t allocatorSize);

		// Updates SVO Tree depending on the changes of the allocators
		double									UpdateSVO();
		
		// Traces entire scene with the given ray params
		// Writes results to intensity texture
		// Uses GBuffer to create inital rays (free first bounce)
		double									ConeTrace(GLuint depthBuffer,
														  GLuint normalBuffer,
														  GLuint colorBuffer,
														  const Camera& camera);

		// Set current scene light positions
		void									LinkScene(GLuint lightBuffer,
														  GLuint shadowMapArrayTexture);

		VoxelDebugVAO							VoxelDataForRendering(double& transferTime,
																	  unsigned int& voxelCount,
																	  unsigned int level);

		uint64_t								MemoryUsage() const;
};

#endif //__GICUDASPARSEVOXELOCTREE_H__
