/**

OGL Context Creation


*/

#ifndef __OGLVOXELIZER_H__
#define __OGLVOXELIZER_H__

#include <array>
#include <IEUtility/IEVector4.h>
#include "GLHeaderLite.h"
#include "GFG/GFGFileLoader.h"
#include "VoxelizerTypes.h"
#include "StructuredBuffer.h"
#include "GFG/GFGFileExporter.h"

#define VOX_PACK_LIMITATION 1024 // Max 1024 voxels can be packed

class MeshBatch;

enum class MipInfo
{
	EMPTY,
	MIP,
	NOT_MIP
};

struct VoxelizerOptions
{
	float		span;
	uint32_t	cascadeCount;
};

#pragma pack(push, 1)
struct ObjInfo
{
	uint32_t voxCount;
	uint32_t voxOffset;
};

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
		StructuredBuffer<ObjInfo>			objectInfos;

		// Batch Related
		StructuredBuffer<uint32_t>			totalVoxCount;

		// Voxel Related
		StructuredBuffer<VoxelPosition>		vPositions;
		StructuredBuffer<VoxelNormal>		vNormals;
		StructuredBuffer<VoxelAlbedo>		vAlbedos;
		StructuredBuffer<VoxelWeights>		vWeights;
		
		// Dense Storage
		GL3DTexture&						lockTex;
		StructuredBuffer<IEVector4>&		vNormalDense;
		StructuredBuffer<IEVector4>&		vAlbedoDense;
		StructuredBuffer<VoxelWeights>&		vWeightDense;

		// Shaders
		Shader&								compSplitCount;
		Shader&								compPackVoxels;
		Shader&								compPackVoxelsSkel;
		
		Shader&								vertVoxelize;
		Shader&								geomVoxelize;
		Shader&								fragVoxelize;
		
		Shader&								vertVoxelizeSkel;
		Shader&								geomVoxelizeSkel;
		Shader&								fragVoxelizeSkel;

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
		double							DetermineSplits(float currentSpan);
		double							AllocateVoxelCaches(bool& hasVoxels, float currentSpan);
		double							GenVoxelWeights();
		double							Voxelize(float currentSpan);
		double							WriteCascadeToGFG(float currentSpan,
														  const std::string& batchName);

		void							VoxelizeObject(uint32_t objIndex, float segmentSize,
													   GLuint splitX, GLuint splitY, GLuint splitZ,
													   float currentSpan);
		void							PackObjectVoxels(uint32_t objIndex,
														 uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ,
														 uint32_t splitX, uint32_t splitY, uint32_t splitZ);
		
		// Size Related
		uint64_t						VoxelSize();

	protected:
	public:
		// Constructors & Destructor
										OGLVoxelizer(const VoxelizerOptions&,
													 MeshBatch&,
													 GL3DTexture& lockTex,
													 StructuredBuffer<IEVector4>& vNormalDense,
													 StructuredBuffer<IEVector4>& vAlbedoDense,
													 StructuredBuffer<VoxelWeights>& vWeightDense,
													 Shader& compSplitCount,
													 Shader& compPackVoxels,
													 Shader& compPackVoxelsSkel,
													 Shader& vertVoxelize,
													 Shader& geomVoxelize,
													 Shader& fragVoxelize,
													 Shader& vertVoxelizeSkel,
													 Shader& geomVoxelizeSkel,
													 Shader& fragVoxelizeSkel,
													 Shader& fragVoxelizeCount,
													 bool isSkeletal);
										~OGLVoxelizer();


		// Voxelization Functions
		void							Execute(const std::string& batchName);

		// Generic Init
		static bool						InitGLSystem();
		static void						DestroyGLSystem();



};
#endif //__OGLVOXELIZER_H__