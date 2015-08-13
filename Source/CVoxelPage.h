/**
*/
#ifndef __CVOXELPAGE_H__
#define __CVOXELPAGE_H__

#define GI_PAGE_SIZE 65536
#define GI_THREAD_PER_BLOCK 256
#define GI_BLOCK_PER_PAGE (GI_PAGE_SIZE / GI_THREAD_PER_BLOCK)
#define GI_SEGMENT_SIZE (GI_BLOCK_PER_PAGE * 4)
#define GI_SEGMENT_PER_PAGE (GI_PAGE_SIZE / GI_SEGMENT_SIZE)

static_assert(GI_PAGE_SIZE % GI_THREAD_PER_BLOCK == 0, "Page size must be divisible by thread per block");

struct CVoxelPage
{
	CVoxelPacked*		dGridVoxels;
	unsigned int*		dEmptySegmentPos;
	char*				dIsSegmentOccupied;
	unsigned int		dEmptySegmentIndex;
	
};
#endif //__CVOXELPAGE_H__