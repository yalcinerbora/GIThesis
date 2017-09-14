#pragma once

// Generic Binding Points
// Textures (Bind Uniforms)
#define T_IN 0
#define T_COLOR 0
#define T_NORMAL 1
#define T_EDGE 1
#define T_DEPTH 2
#define T_SHADOW 3
#define T_INTENSITY 3
#define T_SHADOW_DIR 4
#define T_DENSE_NODE 5
#define T_DENSE_MAT 6

#define I_OUT_TEXTURE 0
#define I_COLOR_FB 2
#define I_VOX_READ 2
#define I_VOX_WRITE 2
#define I_DEPTH_READ 0
#define I_DEPTH_WRITE 1

#define U_RENDER_TYPE 0
#define U_SHADOW_MIP_COUNT 0
#define U_TRESHOLD 0
#define U_DIRECTION 0
#define U_MAX_DISTANCE 0
#define U_DEPTH_SIZE 0
#define U_DO_AO 0
#define U_DO_GI 1
#define U_NEAR_FAR 1
#define U_SPAN 1
#define U_SHADOW_MAP_WH 1
#define U_CONE_ANGLE 1
#define U_NEAR_FAR 1
#define U_FETCH_LEVEL 1
#define U_PIX_COUNT 1
#define U_SAMPLE_DISTANCE 2
#define U_DRAW_ID 2
#define U_LIGHT_INDEX 2
#define U_CAST_SPECULAR_CONE 2
#define U_IMAGE_SIZE 3
#define U_TOTAL_VOX_DIM 3
#define U_OBJ_ID 4
#define U_TOTAL_OBJ_COUNT 4
#define U_MIN_SPAN 5
#define U_MAX_CACHE_SIZE 5
#define U_MAX_GRID_DIM 6
#define U_OBJ_TYPE 6
#define U_IS_MIP 7
#define U_LIGHT_ID 4
#define U_RENDER_MODE 2

// Unfiorm
#define U_FTRANSFORM 0
#define U_INVFTRANSFORM 1
#define U_VOXEL_GRID_INFO 2
#define U_OCTREE_UNIFORMS 3
#define U_INDIRECT_UNIFORMS 4
#define U_GRID_TRANSFORM 5

// Large Uniform
#define LU_LIGHT_MATRIX 0
#define LU_VOXEL_NORM_POS 0
#define LU_VOXEL_RENDER 1
#define LU_LIGHT 1
#define LU_OBJECT_GRID_INFO 2
#define LU_VOXEL_GRID_INFO 2
#define LU_SVO_NODE 2
#define LU_SVO_ILLUM 3
#define LU_SVO_LEVEL_OFFSET 4
#define LU_VOXEL_IDS 3
#define LU_AABB 3
#define LU_MTRANSFORM 4
#define LU_INDEX_CHECK 4
#define LU_MTRANSFORM_INDEX 5
#define LU_JOINT_TRANS 6