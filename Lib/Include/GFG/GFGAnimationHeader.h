/**

GFGKeyFrame Structure
GFGAnimationHeaderCore Structure
GFGAnimationHeader Structure

Animation related structures used by GFGHeader

For License refer to:
https://github.com/yalcinerbora/GFGFileFormat/blob/master/LICENSE
*/

#ifndef __GFG_ANIMATIONHEADER_H__
#define __GFG_ANIMATIONHEADER_H__

#include <vector>
#include "GFGEnumerations.h"

// Pack Those Structs 
#pragma pack(push, 1)

struct GFGKeyFrame
{
	// Time
	double		time;			// In seconds

	// Tangent
	float		inTangentX; 
	float		inTangentY;
	float		outTangentX;
	float		outTangentY;
};

struct GFGAnimationHeaderCore
{
	uint64_t			dataStart;
	uint32_t			skeletonIndex;
	uint32_t			boneIndex;
	uint32_t			keyframeAmount;
};

struct GFGAnimationHeader
{
	GFGAnimationHeaderCore			headerCore;
	std::vector<GFGKeyFrame>		keyFrames;
};

static_assert (sizeof(GFGAnimationHeaderCore) ==
			   sizeof(uint64_t) * 1
			   + sizeof(uint32_t) * 3,
			   "Animation Header Core Size Mismatch from GFG Definition");

static_assert (sizeof(GFGKeyFrame) ==
			   sizeof(float) * 4
			   + sizeof(double) * 1,
			   "Keyframe Size Mismatch from GFG Definition");

#pragma pack(pop)
#endif //__GFG_ANIMATIONHEADER_H__