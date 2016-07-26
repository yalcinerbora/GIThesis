/**

OGL Context Creation


*/

#ifndef __OGLVOXELIZER_H__
#define __OGLVOXELIZER_H__

#include <array>
#include <IEUtility/IEVector4.h>
#include "GLHeaderLite.h"
#include "GFG/GFGFileLoader.h"
#include "VoxelCacheData.h"
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
	float		splatRatio;
	float		span;
	uint32_t	cascadeCount;
};

#pragma pack(push, 1)
struct ObjInfo
{
	float span;
	uint32_t voxCount;
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

		GFGFileExporter						fileOut;
		bool								isSkeletal;
		VoxelizerOptions					options;
		MeshBatch&							batch;

		// Object(Draw count) Related
		StructuredBuffer<ObjVoxSplit>		split;
		StructuredBuffer<ObjInfo>			objectInfos;

		// Batch Related
		StructuredBuffer<uint32_t>			totalVoxCount;

		// Voxel Related
		StructuredBuffer<VoxelNormPos>		voxelNormPos;
		StructuredBuffer<VoxelColorData>	color;
		StructuredBuffer<VoxelWeightData>	weights;
		StructuredBuffer<VoxelIds>			voxIds;
		
		GL3DTexture&						lockTex;
		StructuredBuffer<IEVector4>&		normalArray;
		StructuredBuffer<IEVector4>&		colorArray;

		Shader&								compSplitCount;
		Shader&								compPackVoxels;
		
		Shader&								vertVoxelize;
		Shader&								geomVoxelize;
		Shader&								fragVoxelize;
		Shader&								fragVoxelizeCount;
		
		std::vector<uint8_t>				totalObjInfos;
		std::vector<MipInfo>				mipInfo;

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
		double							AllocateVoxelCaches(float currentSpan, uint32_t curentCascade);
		double							GenVoxelWeights();
		double							Voxelize(float currentSpan);
		double							FormatToGFG(float currentSpan);

		void							VoxelizeObject(uint32_t objIndex, float segmentSize,
													   GLuint splitX, GLuint splitY, GLuint splitZ,
													   float currentSpan);
		void							PackObjectVoxels(uint32_t objIndex, GLuint isMip,
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
													 StructuredBuffer<IEVector4>& normalArray,
													 StructuredBuffer<IEVector4>& colorArray,
													 Shader& compSplitCount,
													 Shader& compPackVoxels,
													 Shader& vertVoxelize,
													 Shader& geomVoxelize,
													 Shader& fragVoxelize,
													 Shader& fragVoxelizeCount,
													 bool isSkeletal);
										~OGLVoxelizer();


		// Voxelization Functions
		void							Start();
		double							Write(const std::string& fileName);


		// Generic Init
		static bool						InitGLSystem();
		static void						DestroyGLSystem();



};
#endif //__OGLVOXELIZER_H__