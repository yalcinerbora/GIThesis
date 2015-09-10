/**
*/
#ifndef __CVOXELPAGE_H__
#define __CVOXELPAGE_H__

#define GI_PAGE_SIZE 65536
#define GI_THREAD_PER_BLOCK 512
#define GI_BLOCK_PER_PAGE (GI_PAGE_SIZE / GI_THREAD_PER_BLOCK)
#define GI_SEGMENT_SIZE 1024
#define GI_SEGMENT_PER_PAGE (GI_PAGE_SIZE / GI_SEGMENT_SIZE)

static_assert(GI_PAGE_SIZE % GI_THREAD_PER_BLOCK == 0, "Page size must be divisible by thread per block");

struct CVoxelPage
{
	CVoxelNormPos*		dGridVoxNormPos;
	CVoxelIds*			dGridVoxIds;
	unsigned int*		dEmptySegmentPos;
	char*				dIsSegmentOccupied;
	unsigned int		dEmptySegmentStackSize;

};

#endif //__CVOXELPAGE_H__