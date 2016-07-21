/**

OGL Context Creation


*/

#ifndef __OGLVOXELIZER_H__
#define __OGLVOXELIZER_H__

#include <array>
#include "GLHeaderLite.h"
#include "GFG/GFGFileLoader.h"
#include "VoxelCacheData.h"
#include "StructuredBuffer.h"

class MeshBatch;

struct VoxelizerOptions
{
	float		splatRatio;
	float		span;
	uint32_t	cascadeCount;
};

#pragma pack(push, 1)
struct ObjVoxSplit
{
	uint16_t voxSplit[4];
};
#pragma pack(pop)

class GL3DTexture;
class Shader;
struct GLFWwindow;

class OGLVoxelizer
{
	private:
		static GLFWwindow*					window;

		bool								isSkeletal;
		VoxelizerOptions					options;
		MeshBatch&							batch;

		// Object(Draw count) Related
		StructuredBuffer<ObjVoxSplit>		split;
		StructuredBuffer<uint32_t>			objVoxCount;

		// Batch Related
		StructuredBuffer<uint32_t>			totalVoxCount;

		// Voxel Related
		StructuredBuffer<VoxelNormPos>		voxelNormPos;
		StructuredBuffer<VoxelColorData>	color;
		StructuredBuffer<VoxelWeightData>	weights;
		
		GL3DTexture&						lockTex;
		GL3DTexture&						normalTex;
		GL3DTexture&						colorTex;

		Shader&								compSplitCount;
		Shader&								compPackVoxels;
		
		Shader&								vertVoxelize;
		Shader&								geomVoxelize;
		Shader&								fragVoxelize;
		Shader&								fragVoxelizeCount;
		
		// Debug Context Callbacks
		static void						ErrorCallbackGLFW(int, const char*);
		static void __stdcall			OGLCallbackRender(GLenum source,
														  GLenum type,
														  GLuint id,
														  GLenum severity,
														  GLsizei length,
														  const GLchar* message,
														  const void* userParam);

		// Logics
		void							DetermineSplits();
		void							AllocateVoxelCaches();

	protected:
	public:
		// Constructors & Destructor
										OGLVoxelizer(const VoxelizerOptions&,
													 MeshBatch&,
													 GL3DTexture& lockTex,
													 GL3DTexture& normalTex,
													 GL3DTexture& colorTex,
													 Shader& compSplitCount,
													 Shader& compPackVoxels,
													 Shader& vertVoxelize,
													 Shader& geomVoxelize,
													 Shader& fragVoxelize,
													 Shader& fragVoxelizeCount,
													 bool isSkeletal);
										~OGLVoxelizer();


		// Voxelization Functions
		float							Voxelize();
										
		

		// Generic Init
		static bool						InitGLSystem();
		static void						DestroyGLSystem();



};

#endif //__OGLVOXELIZER_H__