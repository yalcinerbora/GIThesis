/**

Sparse voxel octree types that used in cuda funcs

*/

#ifndef __CSVOTYPES_H__
#define __CSVOTYPES_H__

// first int has
// first 24 bit is children index
// last 8 bit used to determine which children is avail
// --
// last 4 byte is used for color
typedef unsigned int CSVONode;
typedef unsigned int CSVOColor;

struct CSVOConstants
{
	unsigned int denseDim;
	unsigned int denseDepth;
	unsigned int totalDepth;
	unsigned int numCascades;
};
#endif //__CSVOTYPES_H__