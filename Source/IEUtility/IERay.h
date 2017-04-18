#pragma once

#include "IEVector3.h"

class IEMatrix4x4;

class IERay
{
	private:
		IEVector3			direction;
		IEVector3			position;

	protected:
	public:
		// Constructors & Destructor
		constexpr			IERay();
		constexpr			IERay(float dX, float dY, float dZ,
								  float pX, float pY, float pZ);
							IERay(const IEVector3& direction, const IEVector3& position);
							IERay(const IEVector3[2]);
							IERay(const IERay&) = default;
							~IERay() = default;

		// Assignment Operators
		IERay&				operator=(const IERay&) = default;
		IERay&				operator=(const IEVector3[2]);

		const IEVector3&	getDirection() const;
		const IEVector3&	getPosition() const;

		// Intersections
		bool				IntersectsSphere(IEVector3& pos,
											 float& t,
											 const IEVector3& sphereCenter, 
											 float sphereRadius) const;
		bool				IntersectsTriangle(IEVector3& baryCoords,
											   float& t,
											   const IEVector3 triCorners[3]) const;
		bool				IntersectsTriangle(IEVector3& baryCoords,
											   float& t,
											   const IEVector3& t0,
											   const IEVector3& t1,
											   const IEVector3& t2) const;
		bool				IntersectsAABB(const IEVector3& min, 
										   const IEVector3& max) const;

		// Utility
		IERay				Reflect(const IEVector3& normal) const;
		IERay&				ReflectSelf(const IEVector3& normal);
		bool				Refract(IERay& out, const IEVector3& normal, float fromMedium, float toMedium) const;
		bool				RefractSelf(const IEVector3& normal, float fromMedium, float toMedium);

		IERay				NormalizeDir() const; 
		IERay&				NormalizeDirSelf();
		IERay				Advance(float) const;
		IERay&				AdvanceSelf(float);
		IERay				Transform(const IEMatrix4x4&) const;
		IERay&				TransformSelf(const IEMatrix4x4&);		
		IEVector3			AdvancedPos(float t) const;
};

// Requirements of IERay
static_assert(std::is_literal_type<IERay>::value == true, "IERay has to be literal type");
static_assert(std::is_trivially_copyable<IERay>::value == true, "IERay has to be trivially copyable");
static_assert(std::is_polymorphic<IERay>::value == false, "IERay should not be polymorphic");
static_assert(sizeof(IERay) == sizeof(float) * 6, "IERay size is not 24 bytes");

constexpr IERay::IERay()
	: direction(1.0f, 0.0f, 0.0f)
	, position(0.0f, 0.0f, 0.0f)
{}

constexpr IERay::IERay(float dX, float dY, float dZ,
					   float pX, float pY, float pZ)
	: direction(dX, dY, dZ)
	, position(pX, pY, pZ)
{}

inline IERay::IERay(const IEVector3& direction, const IEVector3& position)
	: direction(direction)
	, position(position)
{}



inline IERay::IERay(const IEVector3 vecList[])
	: direction(vecList[0])
	, position(vecList[1])
{}

inline IERay& IERay::operator=(const IEVector3 vecList[])
{
	direction = vecList[0];
	position = vecList[1];
	return *this;
}

inline const IEVector3& IERay::getDirection() const
{
	return direction;
}

inline const IEVector3& IERay::getPosition() const
{
	return position;
}

inline bool IERay::IntersectsTriangle(IEVector3& baryCoords, float& t, 
									  const IEVector3 triCorners[3]) const
{
	return IntersectsTriangle(baryCoords, t,
							  triCorners[0],
							  triCorners[1],
							  triCorners[2]);
}

