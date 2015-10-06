/**



*/

#ifndef __GICUDASPARSEVOXELOCTREE_H__
#define __GICUDASPARSEVOXELOCTREE_H__

#include <cuda.h>
#include "CudaVector.cuh"
#include "CSparseVoxelOctree.cuh"
#include "SceneLights.h"

class GICudaAllocator;
class Camera;

class GISparseVoxelOctree
{
	private:
		GICudaAllocator*						allocator;		// Page Allocators
		size_t									allocatorSize;	// Page

		CVoxelGrid								hVoxelGrid;
		CudaVector<CVoxelGrid>					dVoxelGrid;

		// SVO Data
		cudaTextureObject_t						dSVOUpper;	// Up to lvl 64 (64 included)
		std::vector<CudaVector<CSVONode>>		dSVOLower;	// Rest is sparse
		CudaVector<CSVONode*>					dSVOLower2D;

		// Inital Rays buffer
		GLuint									initalRayLink;
		cudaGraphicsResource_t					rayLinks;

		// Interop Data
		cudaGraphicsResource_t					shadowMapArrayLink;
		cudaGraphicsResource_t					lightBufferLink;
		cudaGraphicsResource_t					lightIntensityLink;
		
	protected:

	public:
		// Constructors & Destructor
												GISparseVoxelOctree(GICudaAllocator* allocator, 
																	size_t allocatorSize,
																	GLuint lightIntensityBuffer);
												GISparseVoxelOctree(const GISparseVoxelOctree&) = delete;
		GISparseVoxelOctree&					operator=(const GISparseVoxelOctree&) = delete;
												~GISparseVoxelOctree();

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

};

//// SVO Lower
//for(auto& svoLevel : dSVOLower)
//{
//	memory += svoLevel.Size() * sizeof(CSVONode);
//}
//
//// SVO Upper
//memory += SVOTextureSize *
//SVOTextureSize *
//SVOTextureSize *
//sizeof(unsigned int);



//// Allocate SVO
//std::vector<CSVONode*> hSVOLower2D;
//for(auto& svoLevel : dSVOLower)
//{
//	svoLevel.Resize(pageAmount * GI_PAGE_SIZE);
//	hSVOLower2D.push_back(svoLevel.Data());
//}
//dSVOLower2D = hSVOLower2D;

#endif //__GICUDASPARSEVOXELOCTREE_H__
