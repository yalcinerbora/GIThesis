#pragma once

#include "IEVector3.h"

class IEBoundingSphere
{
	public:
		IEVector3	center;
		float		radius;

		IEBoundingSphere(const IEVector3&, float);
};

// Requirements of Vector3
static_assert(std::is_trivially_copyable<IEBoundingSphere>::value == true, "IEVector3 has to be trivially copyable");
static_assert(std::is_polymorphic<IEBoundingSphere>::value == false, "IEVector3 should not be polymorphic");
static_assert(sizeof(IEBoundingSphere) == sizeof(float) * 4, "IEVector3 size is not 12 bytes");