#include "IEAxisAalignedBB.h"
#include "IERay.h"

static constexpr IEVector3 aabbLookupTable[] =
{
	{ 1.0f, 1.0f, 1.0f },		// V1
	{ 0.0f, 1.0f, 1.0f },		// V2
	{ 1.0f, 0.0f, 1.0f },		// V3
	{ 1.0f, 1.0f, 0.0f },		// V4

	{ 0.0f, 0.0f, 1.0f },		// V5
	{ 0.0f, 1.0f, 0.0f },		// V6
	{ 1.0f, 0.0f, 0.0f },		// V7
	{ 0.0f, 0.0f, 0.0f }		// V8
};

IEAxisAlignedBB3 IEAxisAlignedBB3::Transform(const IEMatrix4x4& t) const
{
	IEVector3 newMin(std::numeric_limits<float>::max());
	IEVector3 newMax(-std::numeric_limits<float>::max());
	for(unsigned int i = 0; i < 8; i++)
	{
		IEVector3 data;
		data[0] = aabbLookupTable[i][0] * gObjectAABB.max.x + (1.0f - aabbLookupTable[i].x) * gObjectAABB.min.x;
		data[1] = aabbLookupTable[i][1] * gObjectAABB.max.y + (1.0f - aabbLookupTable[i].y) * gObjectAABB.min.y;
		data[2] = aabbLookupTable[i][2] * gObjectAABB.max.z + (1.0f - aabbLookupTable[i].z) * gObjectAABB.min.z;

		data = t * data;
		newMax[0] = fmax(newMax[0], data[0]);
		newMax[1] = fmax(newMax[1], data[1]);
		newMax[2] = fmax(newMax[2], data[2]);

		newMin[0] = fmin(newMin[0], data[0]);
		newMin[1] = fmin(newMin[1], data[1]);
		newMin[2] = fmin(newMin[2], data[2]);
	}

	return IEAxisAlignedBB3();
}

IEAxisAlignedBB3& IEAxisAlignedBB3::TransformSelf(const IEMatrix4x4&)
{
	return *this;
}

bool IEAxisAlignedBB3::Intersects(const IERay& ray) const
{
	return ray.IntersectsAABB(min, max);
}

//-----------------------------//
//IEAxisAlignedBB2 IEAxisAlignedBB2::Transform(const IEMatrix4x4&) const
//{
//	return IEAxisAlignedBB2();
//}
//
//IEAxisAlignedBB2& IEAxisAlignedBB2::TransformSelf(const IEMatrix4x4&)
//{
//	return *this;
//}
//
//bool IEAxisAlignedBB2::Intersects(const IERay& ray) const
//{	return false;
//}