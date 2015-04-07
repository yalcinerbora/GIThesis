/**

GFGBone Structure
GFGSkeletonHeader Structure

Skeleton related data structures used by GFGHeader class.

For License refer to:
https://github.com/yalcinerbora/GFGFileFormat/blob/master/LICENSE
*/

#ifndef __GFG_SKELETONHEADER_H__
#define __GFG_SKELETONHEADER_H__

#include <vector>
#include "GFGEnumerations.h"

// Pack Those Structs 
#pragma pack(push, 1)


// Components of this material
struct GFGBone
{
	uint32_t	parentIndex;		// This index is for this array
	uint32_t	transformIndex;		// Transform of the bind pose
};

struct GFGSkeletonHeader
{
	uint32_t				boneAmount;
	std::vector<GFGBone>	bones;
};

static_assert (sizeof(GFGBone) == sizeof(uint32_t) * 2,
			   "Bone Size Mismatch from GFG Definition");

#pragma pack(pop)
#endif //__GFG_SKELETONHEADER_H__