#pragma once

#define LU_AABB layout(std430, binding = 3) readonly
#define LU_NORMAL_DENSE layout(std430, binding = 6) coherent volatile
#define LU_ALBEDO_DENSE layout(std430, binding = 7) coherent volatile
#define LU_VOXEL_DATA layout(std430, binding = 8) coherent volatile
