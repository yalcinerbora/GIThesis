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

enum class GFGQuatInterpType : uint16_t
{
	SLERP,
	CUBIC
};

enum class GFGQuatLayout : uint16_t
{
	WXYZ,
	XYZW
};

enum class GFGAnimType : uint32_t
{
	WITH_HIP_TRANSLATE,
	ONLY_QUATERNION
};

enum class GFGAnimationLayout : uint32_t
{
	// M is number of keys
	// N is number of joints
	// T is key time (in seconds)
	// HT is hip translate
	// If has translation, those values will be available
	KEYS_OF_BONES,	// (T0 --- TM), B0(HT0 --- HTM), B0(K0 --- KM), B1(K0 --- KM), .... BN(K0 --- KM)
	BONES_OF_KEYS	// K0(T0, HT0, B0 --- BN), K1(T1, HT1, B0 --- BN), .... KM(TM, HTM, B0 --- BN)
};

struct GFGAnimationHeader
{
	uint64_t					dataStart;
	GFGAnimationLayout			layout;
	GFGAnimType					type;
	GFGQuatInterpType			interpType;
	GFGQuatLayout				quatType;
	uint32_t					skeletonIndex;
	uint32_t					keyCount;
};

// Confirm that it is packed
static_assert (sizeof(GFGAnimationHeader) ==
				 sizeof(uint64_t) * 1
			   + sizeof(uint32_t) * 2
			   + sizeof(GFGAnimationLayout)
			   + sizeof(GFGAnimType)
			   + sizeof(GFGQuatInterpType)
			   + sizeof(GFGQuatLayout),
			   "Animation Header Core Size Mismatch from GFG Definition");

#pragma pack(pop)
#endif //__GFG_ANIMATIONHEADER_H__