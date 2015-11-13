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
		CudaVector<CSVOMaterial>				dSVOMaterial;			// Entire SVO

		// SVO Ptrs
		CSVONode*								dSVODense;
		CSVONode*								dSVOSparse;

		// SVO Mat indices
		uint32_t								matSparseOffset;

		// Atomic counter and svo level start locations
		std::vector<CudaVector<uint32_t>>		dSVONodeId;
		CudaVector<unsigned int>				dSVONodeAllocator;
		CudaVector<unsigned int>				dSVOLevelSizes;
		std::vector<unsigned int>				hSVOLevelSizes;
		std::vector<unsigned int>				hSVOLevelOffsets;

		// Inital Rays buffer
		GLuint									initalRayLink;
		cudaGraphicsResource_t					rayLinks;

		// Interop Data
		cudaGraphicsResource_t					shadowMapArrayTexLink;
		cudaGraphicsResource_t					lightBufferLink;
		cudaGraphicsResource_t					lightIntensityTexLink;
		
		void									ConstructDense();
		void									ConstructLevel(unsigned int levelIndex,
															   unsigned int allocatorIndex);
		void									ConstructFullAtomic();
		void									ConstructLevelByLevel();
		void									AverageNodes(bool orderedNodes);

	protected:

	public:
		// Constructors & Destructor
												GISparseVoxelOctree(GLuint lightIntensityTex);
												GISparseVoxelOctree(const GISparseVoxelOctree&) = delete;
		GISparseVoxelOctree&					operator=(const GISparseVoxelOctree&) = delete;
												~GISparseVoxelOctree();

		// Link Allocators and Adjust Size of the System
		void									 LinkAllocators(GICudaAllocator** newAllocators,
																size_t allocatorSize,
																float sceneMultiplier);

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

		double									SVODataToGL(// GL buffer ptrs
															CVoxelNormPos* dVAONormPosData,
															uint32_t* dVAOColorData,

															CVoxelGrid& voxGridData,
															uint32_t& voxCount,
															uint32_t level,
															uint32_t maxVoxelCount);

		uint64_t								MemoryUsage() const;
};
#endif //__GICUDASPARSEVOXELOCTREE_H__
