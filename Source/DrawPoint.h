/**

GPU Buffer That Holds Mesh (and Index)

Draw Points (Aka Draw Call Parameters) in that buffer
Draw Points Struct Applicable to the Indirect Command Buffer in OGL

*/

#ifndef __DRAWPOINT_H__
#define __DRAWPOINT_H__

#pragma pack(push, 1)
struct DrawPoint
{
	uint32_t	count;
	uint32_t	instanceCount;
	uint32_t	baseVertex;
	uint32_t	baseInstance;
};

struct DrawPointIndexed
{
	uint32_t	count;
	uint32_t	instanceCount;
	uint32_t	firstIndex;
	uint32_t	baseVertex;
	uint32_t	baseInstance;
};
#pragma pack(pop)

static_assert(sizeof(DrawPointIndexed) == 20, "Size of DrawPointIndexed is not 20 byte");
static_assert(sizeof(DrawPoint) == 16, "Size of DrawPoint is not 16 byte");

#endif //__DRAWPOINT_H__