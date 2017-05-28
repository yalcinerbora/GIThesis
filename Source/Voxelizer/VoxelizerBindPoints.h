#ifndef __BINDPOINTS_H__
#define __BINDPOINTS_H__

#define BLOCK_SIZE 256

// Shader Storage
#define LU_VOXEL_POS 0
#define LU_VOXEL_NORM 1
#define LU_VOXEL_ALBEDO 2
#define LU_VOXEL_WEIGHT 4

#define LU_AABB 3
#define LU_INDEX_ATOMIC 5
#define LU_MESH_VOXEL_INFO 2
#define LU_MESH_SPLIT_INFO 4
#define LU_TOTAL_VOX_COUNT 4

#define LU_NORMAL_DENSE 6
#define LU_ALBEDO_DENSE 7
#define LU_WEIGHT_DENSE 8

// Image
#define I_LOCK 0

// Textures
#define T_COLOR 0
#define T_NORMAL 1
#define T_SPECULAR 2

// Uniform Bufer

// Uniform
#define U_TOTAL_OBJ_COUNT 0
#define U_SPAN 1
#define U_GRID_DIM 2
#define U_VOX_LIMIT 3
#define U_TOTAL_VOX_DIM 3
#define U_SEGMENT_SIZE 2
#define U_OBJ_ID 4
#define U_SPLAT_RATIO 5
#define U_MAX_CACHE_SIZE 5
#define U_OBJ_TYPE 6
#define U_SPLIT_CURRENT 7
#define U_TEX_SIZE 8
#define U_IS_MIP 9

#endif //__BINDPOINTS_H__