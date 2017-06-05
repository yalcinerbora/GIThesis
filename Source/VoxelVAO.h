#pragma once
/**

Vao for rendering voxel debug

*/


#include "GLHeaderLite.h"
#include <cstdint>
#include <vector_types.h>
#include <array>
#include "VoxelizerTypes.h"
#include "StructuredBuffer.h"

#define IN_CUBE_POS 0
#define IN_VOXEL_POS 1
#define IN_VOXEL_NORM 2
#define IN_VOXEL_ALBEDO 3
#define IN_VOXEL_WEIGHT 4

class VoxelVAO
{
	public:
		static constexpr char*			CubeGFGFileName = "cube.gfg";		
		struct CubeOGL
		{
			uint32_t					drawCount;
			std::vector<uint8_t>		data;
		};

	private:
		GLuint							vao;

	public:
		// Cosntructors & Destructor
										VoxelVAO();
										VoxelVAO(StructuredBuffer<uint8_t>& buffer,
												 size_t cubePosOffset,
												 size_t voxPosOffset,
												 size_t voxNormOffset,
												 size_t voxAlbedoOffset = 0,
												 size_t voxWeightOffset = 0);
										VoxelVAO(const VoxelVAO&) = delete;
										VoxelVAO(VoxelVAO&&);
		VoxelVAO&						operator=(const VoxelVAO&) = delete;
		VoxelVAO&						operator=(VoxelVAO&&);
										~VoxelVAO();

		void							Bind();
		void							Draw(uint32_t cubeIndexSize,
											 uint32_t voxelCount, 
											 uint32_t offset);
		void							Draw(uint32_t parameterOffset);

		static CubeOGL					LoadCubeDataFromGFG();
};