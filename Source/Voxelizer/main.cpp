
#include <string>
#include <sstream>
#include <iostream>
#include <GFG/GFGFileLoader.h>

#include "OGLVoxelizer.h"
#include "MeshBatchSkeletal.h"
#include "GL3DTexture.h"
#include "Shader.h"
#include "Macros.h"
#include "VoxFramebuffer.h"
#include "Globals.h"

static const VoxelizerOptions defaults =
{
	1.0f,		// 1 pixel per voxel
	1.0f,		// 1 unit coverage per voxel
//	512,		// 512^3 voxel dim
	1			// Single cascade
};

enum class VoxErrorType
{
	OK,
	PARSE_ERROR
};

const std::string switches[] = 
{
	"-f",		// File	(n string arguments)
	"-fs",		// Skeletal File (n string arguments)
	"-span",	// Voxel Size (1 float argument)
	"-cas",		// Cascade count (1 int argument)
	"-splat"	// Splat ratio (1 float argument)
};

bool isSwitch(const std::string& switchNominee)
{
	for(const std::string& sw : switches)
		if(switchNominee == sw) return true;
	return false;
}

VoxErrorType ParseOptions(VoxelizerOptions& opts, 
						  std::vector<std::string>& fileNames, 
						  std::vector<std::string>& fileNamesSkel,
						  int argc, char* argv[])
{
	bool foundFile = false;
	// Consume Args
	for(int i = 1; i < argc; i++)
	{
		const std::string arg = argv[i];

		int argId = 0;
		for(const std::string& sw : switches)
		{
			if(arg == sw)
			{
				if(argId == 0)	// -f
				{
					// Consume Files until next valid switch
					int j = i + 1;
					while(j < argc && !isSwitch(argv[j]))
					{
						fileNames.emplace_back(argv[j]);
						j++;
					}
					if(j == (i + 1))
					{
						std::cout << "-f switch needs at least one file name" << std::endl;
						return VoxErrorType::PARSE_ERROR;
					}
					i = j - 1;
					break;
				}
				else if(argId == 1) // -fs
				{
					// Consume Files until next valid switch
					int j = i + 1;
					while(j < argc && !isSwitch(argv[j]))
					{
						fileNamesSkel.emplace_back(argv[j]);
						j++;
					}
					if(j == (i + 1))
					{
						std::cout << "-fs switch needs at least one file name" << std::endl;
						return VoxErrorType::PARSE_ERROR;
					}
					i = j - 1;
					break;
				}
				else if(argId == 2)	// -span
				{
					i++;
					if(i < argc && !isSwitch(argv[i]))
					{
						opts.span = std::stof(argv[i]);
					}
					else
					{
						std::cout << "-span switch needs at least one float value" << std::endl;
						return VoxErrorType::PARSE_ERROR;
					}
					break;
				}
				else if(argId == 3) // -cas
				{
					i++;
					if(i < argc && !isSwitch(argv[i]))
					{
						opts.cascadeCount = std::stoi(argv[i]);
					}
					else
					{
						std::cout << "-cas switch needs at least one int value" << std::endl;
						return VoxErrorType::PARSE_ERROR;
					}
					break;
				}
				else if(argId == 4) // -splat
				{
					i++;
					if(i < argc && !isSwitch(argv[i]))
					{
						opts.splatRatio = std::stof(argv[i]);
					}
					else
					{
						std::cout << "-splat switch needs at least one float value" << std::endl;
						return VoxErrorType::PARSE_ERROR;
					}
					break;
				}
			}
			argId++;
		}
	}
	return VoxErrorType::OK;
}

int main(int argc, char* argv[])
{
	std::vector<std::string> fileNames;
	std::vector<std::string> skeletalFileNames;

	VoxelizerOptions options = defaults;
	VoxErrorType error = ParseOptions(options, 
									  fileNames,
									  skeletalFileNames,
									  argc, argv);
	
	if(error != VoxErrorType::OK) return 1;
	if(!OGLVoxelizer::InitGLSystem()) return 1;
	
	GL3DTexture lockTex(TextureDataType::UINT_1);
	StructuredBuffer<IEVector4> normalArray(VOX_3D_TEX_SIZE *
											VOX_3D_TEX_SIZE *
											VOX_3D_TEX_SIZE);
	StructuredBuffer<IEVector4> colorArray(VOX_3D_TEX_SIZE *
										   VOX_3D_TEX_SIZE *
										   VOX_3D_TEX_SIZE);
	StructuredBuffer<VoxelWeightData> weightArray(VOX_3D_TEX_SIZE *
												  VOX_3D_TEX_SIZE *
												  VOX_3D_TEX_SIZE);

	Shader compSplitCount(ShaderType::COMPUTE, "Shaders/DetermineSplitCount.glsl");
	Shader compPackVoxels(ShaderType::COMPUTE, "Shaders/PackObjectVoxels.glsl");
	Shader compPackVoxelsSkel(ShaderType::COMPUTE, "Shaders/PackObjectVoxelsSkel.glsl");

	Shader vertVoxelize(ShaderType::VERTEX, "Shaders/VoxelizeGeom.vert");
	Shader geomVoxelize(ShaderType::GEOMETRY, "Shaders/VoxelizeGeom.geom");
	Shader fragVoxelize(ShaderType::FRAGMENT, "Shaders/VoxelizeGeom.frag");

	Shader vertVoxelizeSkel(ShaderType::VERTEX, "Shaders/VoxelizeGeomSkel.vert");
	Shader geomVoxelizeSkel(ShaderType::GEOMETRY, "Shaders/VoxelizeGeomSkel.geom");
	Shader fragVoxelizeSkel(ShaderType::FRAGMENT, "Shaders/VoxelizeGeomSkel.frag");

	Shader fragVoxelizeCount(ShaderType::FRAGMENT, "Shaders/VoxelizeGeomCount.frag");
	
	// FBO
	GLsizei frameSize = static_cast<GLsizei>(VOX_3D_TEX_SIZE * options.splatRatio);
	VoxFramebuffer fbo(frameSize, frameSize);
	fbo.Bind();

	// Voxelization
	float span = options.span;
	std::stringstream voxPrefix;
	voxPrefix << "vox_" << span << "_" << options.cascadeCount << "_";

	for(auto& fileName : fileNames)
	{
		GI_LOG("");
		GI_LOG("#######################################");
		GI_LOG("Working on \"%s\"...", fileName.c_str());
		MeshBatch batch(rigidMeshVertexDefinition, sizeof(VAO),
						{fileName});
		GI_LOG("");

		OGLVoxelizer voxelizer(options,
							   batch,
							   lockTex,
							   normalArray,
							   colorArray,
							   weightArray,
							   compSplitCount,
							   compPackVoxels,
							   compPackVoxelsSkel,
							   vertVoxelize,
							   geomVoxelize,
							   fragVoxelize,
							   vertVoxelizeSkel,
							   geomVoxelizeSkel,
							   fragVoxelizeSkel,
							   fragVoxelizeCount,
							   false);

		voxelizer.Start();

		std::string voxFile = voxPrefix.str() + fileName;
		voxelizer.Write(voxFile);
	}

	// Skeletal Voxelization
	for(auto& fileName : skeletalFileNames)
	{
		GI_LOG("");
		GI_LOG("#######################################");
		GI_LOG("Working on \"%s\"...", fileName.c_str());		
		MeshBatchSkeletal batch(skeletalMeshVertexDefinition, sizeof(VAOSkel),
							    {fileName});
		GI_LOG("");

		OGLVoxelizer voxelizer(options,
							   batch,
							   lockTex,
							   normalArray,
							   colorArray,
							   weightArray,
							   compSplitCount,
							   compPackVoxels,
							   compPackVoxelsSkel,
							   vertVoxelize,
							   geomVoxelize,
							   fragVoxelize,
							   vertVoxelizeSkel,
							   geomVoxelizeSkel,
							   fragVoxelizeSkel,
							   fragVoxelizeCount,
							   true);
			voxelizer.Start();
			std::string voxFile = voxPrefix.str() + fileName;
			voxelizer.Write(voxFile);
	}
	OGLVoxelizer::DestroyGLSystem();
	return 0;
}