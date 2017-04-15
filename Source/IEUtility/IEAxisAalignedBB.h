#pragma once

#include "IEVector3.h"
#include "IEVector2.h"

class IEMatrix4x4;
class IERay;

class IEAxisAlignedBB3
{
	private:
		static constexpr int	Dimension = 3;

		IEVector3				min;
		IEVector3				max;

	public:
		// Constructors & Destructor
		constexpr				IEAxisAlignedBB3();
		constexpr				IEAxisAlignedBB3(float xMin, float xMax,
												 float yMin, float yMax,
												 float zMin, float zMax);
								IEAxisAlignedBB3(const IEVector2& x, const IEVector2& y, const IEVector2& z);
								IEAxisAlignedBB3(const IEVector3& min, const IEVector3& max);
								IEAxisAlignedBB3(const IEVector2 v[Dimension]);
								IEAxisAlignedBB3(const IEVector3 mm[2]);
								IEAxisAlignedBB3(const IEAxisAlignedBB3&) = default;
								~IEAxisAlignedBB3() = default;

		// Functions
		IEAxisAlignedBB3		Transform(const IEMatrix4x4&) const;
		IEAxisAlignedBB3&		TransformSelf(const IEMatrix4x4&);

		bool					Intersects(const IERay& ray) const;
		
		IEVector3				Min() const;
		IEVector3				Max() const;
};

// Requirements of AABB3
static_assert(std::is_literal_type<IEAxisAlignedBB3>::value == true, "IEAxisAlignedBB3 has to be literal type");
static_assert(std::is_trivially_copyable<IEAxisAlignedBB3>::value == true, "IEAxisAlignedBB3 has to be trivially copyable");
static_assert(std::is_polymorphic<IEAxisAlignedBB3>::value == false, "IEAxisAlignedBB3 should not be polymorphic");
static_assert(sizeof(IEAxisAlignedBB3) == sizeof(float) * 6, "IEAxisAlignedBB3 size is not 24 bytes");

constexpr IEAxisAlignedBB3::IEAxisAlignedBB3()
	: min(0.0f, 0.0f, 0.0f)
	, max(0.0f, 0.0f, 0.0f)
{}

constexpr IEAxisAlignedBB3::IEAxisAlignedBB3(float xMin, float xMax,
											 float yMin, float yMax,
											 float zMin, float zMax)
	: min(xMin, yMin, zMin)
	, max(xMax, yMax, zMax)
{}

inline IEAxisAlignedBB3::IEAxisAlignedBB3(const IEVector2& x, const IEVector2& y, const IEVector2& z)
	: min(x[0], y[0], z[0])
	, max(x[1], y[1], z[1])
{}

inline IEAxisAlignedBB3::IEAxisAlignedBB3(const IEVector3& min, const IEVector3& max)
	: min(min)
	, max(max)
{}

inline IEAxisAlignedBB3::IEAxisAlignedBB3(const IEVector2 v[Dimension])
	: min(v[0][0], v[1][0], v[2][0])
	, max(v[0][1], v[1][1], v[2][1])
{}

inline IEAxisAlignedBB3::IEAxisAlignedBB3(const IEVector3 mm[2])
	: min(mm[0])
	, max(mm[1])
{}

inline IEVector3 IEAxisAlignedBB3::Min() const { return min; }
inline IEVector3 IEAxisAlignedBB3::Max() const { return max; }

//class IEAxisAlignedBB2
//{
//	private:
//	static constexpr int	Dimension = 2;
//
//	IEVector2				min;
//	IEVector2				max;
//
//
//	public:
//	// Constructors & Destructor
//	constexpr				IEAxisAlignedBB2();
//	constexpr				IEAxisAlignedBB2(float xMin, float xMax,
//											 float yMin, float yMax);
//							IEAxisAlignedBB2(const IEVector2& x, const IEVector2& y);
//							IEAxisAlignedBB2(const IEVector2 v[Dimension]);
//							IEAxisAlignedBB2(const IEAxisAlignedBB2&) = default;
//							~IEAxisAlignedBB2() = default;
//
//	// Functions
//	IEAxisAlignedBB2		Transform(const IEMatrix4x4&) const;
//	IEAxisAlignedBB2&		TransformSelf(const IEMatrix4x4&);
//
//	bool					Intersects(const IERay& ray) const;
//};
//
//// Requirements of AABB2
//static_assert(std::is_literal_type<IEAxisAlignedBB2>::value == true, "IEAxisAlignedBB2 has to be literal type");
//static_assert(std::is_trivially_copyable<IEAxisAlignedBB2>::value == true, "IEAxisAlignedBB2 has to be trivially copyable");
//static_assert(std::is_polymorphic<IEAxisAlignedBB2>::value == false, "IEAxisAlignedBB2 should not be polymorphic");
//static_assert(sizeof(IEAxisAlignedBB2) == sizeof(float) * 4, "IEAxisAlignedBB2 size is not 16 bytes");
//
//constexpr IEAxisAlignedBB2::IEAxisAlignedBB2()
//	: mmX(0.0f, 0.0f)
//	, mmY(0.0f, 0.0f)
//{}
//
//constexpr IEAxisAlignedBB2::IEAxisAlignedBB2(float xMin, float xMax,
//											 float yMin, float yMax)
//	: mmX(xMin, xMax)
//	, mmY(yMin, yMax)
//{}
//
//inline IEAxisAlignedBB2::IEAxisAlignedBB2(const IEVector2& x, const IEVector2& y)
//	: mmX(x)
//	, mmY(y)
//{}
//
//inline IEAxisAlignedBB2::IEAxisAlignedBB2(const IEVector2 v[Dimension])
//	: mmX(v[0])
//	, mmY(v[1])
//
//{}