/**

Sparse voxel octree types that used in cuda funcs

*/

#ifndef __CSVOTYPES_H__
#define __CSVOTYPES_H__

#define GI_DENSE_WORKER_PER_PARENT 8

#include <cstdint>

// first int has
// first 32 bit is children index
typedef unsigned int CSVONode;
typedef unsigned int CSVOColor;
typedef unsigned int CSVOLight;

typedef uint64_t CSVOMaterial;

//struct CSVOMaterial
//{
//	unsigned int color;		
//	unsigned int normal;
//	// Only Holding Normal and Color
//
//
//	//unsigned int props;
//	
//	// Center Nodes Has the Layout as
//	// color 8 bit each channel (RGB) A channel is empty (Pre-multiplied Alpha averaged)
//	// normal first 16 bit X, remaining 15 bit is Y, 1 bit is Z sign
//	// props first 8 bit roughness, other 8 bit is metalicity,
//	// Last 16 bit used for directional opacity
//	// 
//
//	// Leaf node has diferent values 
//	// since it will be atomically updated 
//	// color is same A channel holds count
//	// normal is same it will be updated atomiccally using count in the color component
//	// props first 16 bits same last 8 bit is count
//	// directional opacity is always opaque so its omitted
//
//	// updating structure require two updates 
//	// 64-bit atomic update for color and normal)
//	// 32-bit atomic update for material data
//};
//
//struct CSVOLeaf
//{
//};

struct CSVOConstants
{
	unsigned int denseDim;
	unsigned int denseDepth;
	unsigned int totalDepth;
	unsigned int numCascades;
};
#endif //__CSVOTYPES_H__