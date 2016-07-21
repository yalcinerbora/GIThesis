#ifndef __BINDPOINTS_H__
#define __BINDPOINTS_H__

#define BLOCK_SIZE 256

// Shader Storage
#define LU_OBJECT_VOXEL_INFO 2
#define LU_AABB 3
#define LU_OBJECT_SPLIT_INFO 4
#define LU_TOTAL_VOX_COUNT 4

// Image
#define I_LOCK 0
#define I_NORMPOS 1
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
#define U_OBJ_ID 4

#endif //__BINDPOINTS_H__