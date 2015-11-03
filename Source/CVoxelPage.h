/**
*/
#ifndef __CVOXELPAGE_H__
#define __CVOXELPAGE_H__

#include <vector_types.h>

// Used in kernel calls that may not saturate enough cores
#define GI_THREAD_PER_BLOCK_SMALL 64

#define GI_PAGE_SIZE 65536
#define GI_THREAD_PER_BLOCK 512
#define GI_BLOCK_PER_PAGE (GI_PAGE_SIZE / GI_THREAD_PER_BLOCK)
#define GI_SEGMENT_SIZE 1024
#define GI_SEGMENT_PER_PAGE (GI_PAGE_SIZE / GI_SEGMENT_SIZE)
#define GI_BLOCK_PER_SEGMENT (GI_SEGMENT_SIZE / GI_THREAD_PER_BLOCK)

#define GI_THREAD_PER_BLOCK_PRIME 521

static_assert(GI_PAGE_SIZE % GI_THREAD_PER_BLOCK == 0, "Page size must be divisible by thread per block");
static_assert(GI_BLOCK_PER_SEGMENT != 0, "Segment should be bigger(or equal) than block");
static_assert(GI_THREAD_PER_BLOCK_PRIME - GI_THREAD_PER_BLOCK == 9, "ThreadPerBlock and its prime does not seem to be related");

typedef uint2 CVoxelNormPos;
typedef uint2 CVoxelIds;

enum class SegmentOccupation : unsigned char
{
	EMPTY = 0,
	OCCUPIED = 1,
	MARKED_FOR_CLEAR = 2,
};

struct CVoxelPage
{
	CVoxelNormPos*		dGridVoxNormPos;
	CVoxelIds*			dGridVoxIds;
	unsigned char*		dEmptySegmentPos;
	SegmentOccupation*	dIsSegmentOccupied;
	unsigned int		dEmptySegmentStackSize;

};

#endif //__CVOXELPAGE_H__