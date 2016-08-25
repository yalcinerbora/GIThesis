#ifndef __BINDPOINTS_H__
#define __BINDPOINTS_H__

#define BLOCK_SIZE 256

// Shader Storage
#define LU_VOXEL_NORM_POS 0
#define LU_VOXEL_COLOR 1
#define LU_OBJECT_VOXEL_INFO 2
#define LU_AABB 3
#define LU_OBJECT_SPLIT_INFO 4
#define LU_TOTAL_VOX_COUNT 4
#define LU_INDEX_CHECK 4
#define LU_VOXEL_IDS 5
#define LU_NORMAL_SPARSE 6
#define LU_COLOR_SPARSE 7
#define LU_WEIGHT_SPARSE 8
#define LU_VOXEL_WEIGHT 9

// Image
#define I_LOCK 0
#define I_NORMAL 1
#define I_COLOR 2

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