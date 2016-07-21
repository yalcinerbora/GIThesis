
#include <string>
#include <iostream>
#include <GFG/GFGFileLoader.h>

#include "OGLVoxelizer.h"
#include "MeshBatchSkeletal.h"
#include "GL3DTexture.h"
#include "Shader.h"
#include "VoxFramebuffer.h"

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
	
	GL3DTexture lockTex(TextureDataType::BYTE_1);
	GL3DTexture normalTex(TextureDataType::FLOAT_4);
	GL3DTexture colorTex(TextureDataType::FLOAT_4);

	Shader compSplitCount(ShaderType::COMPUTE, "Shaders/DetermineSplitCount.glsl");
	Shader compPackVoxels(ShaderType::COMPUTE, "Shaders/PackObjectVoxels.glsl");

	Shader vertVoxelize(ShaderType::VERTEX, "Shaders/VoxelizeGeom.vert");
	Shader geomVoxelize(ShaderType::GEOMETRY, "Shaders/VoxelizeGeom.geom");
	Shader fragVoxelize(ShaderType::FRAGMENT, "Shaders/VoxelizeGeom.frag");
	Shader fragVoxelizeCount(ShaderType::FRAGMENT, "Shaders/VoxelizeGeomCount.frag");
	
	// FBO
	GLsizei frameSize = static_cast<GLsizei>(VOX_3D_TEX_SIZE * options.splatRatio);
	VoxFramebuffer fbo(frameSize, frameSize);
	fbo.Bind();

	// Voxelization
	for(auto& fileName : fileNames)
	{
		MeshBatch batch(fileName.c_str(), 0.0f, {nullptr, 0}, false);

		for(unsigned int i = 0; i < options.cascadeCount; i++)
		{
			options.span = options.span * static_cast<float>(1 << i);
			OGLVoxelizer voxelizer(options,
								   batch,
								   lockTex,
								   normalTex,
								   colorTex,
								   compSplitCount,
								   compPackVoxels,
								   vertVoxelize,
								   geomVoxelize,
								   fragVoxelize,
								   fragVoxelizeCount,
								   false);

			voxelizer.Voxelize();
			//voxelizer.Write(fileName)
		}
	}

	// Skeletal Voxelization
	for(auto& fileName : skeletalFileNames)
	{
		MeshBatchSkeletal batch(fileName.c_str(), 0.0f, {nullptr, 0});

		for(unsigned int i = 0; i < options.cascadeCount; i++)
		{
			options.span = options.span * static_cast<float>(1 << i);
			OGLVoxelizer voxelizer(options,
								   batch,
								   lockTex,
								   normalTex,
								   colorTex,
								   compSplitCount,
								   compPackVoxels,
								   vertVoxelize,
								   geomVoxelize,
								   fragVoxelize,
								   fragVoxelizeCount,
								   true);

			voxelizer.Voxelize();
			//voxelizer.Write(fileName)
		}
	}


	OGLVoxelizer::DestroyGLSystem();
	return 0;
}