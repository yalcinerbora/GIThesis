#pragma once
/**

Column Major Vector (3x1 Matrix) (NOT 1x3)

*/
#include <algorithm>
#include <cassert>

class IEVector4;
class IEVector3
{
	private:
		static constexpr int	VectorW = 3;
		union
		{
			struct				{float x, y, z;};
			float				v[VectorW];
		};

	protected:

	public:
		// Constructors & Destructor
		constexpr				IEVector3();
		constexpr				IEVector3(float xyz);
		constexpr				IEVector3(float x, float y, float z);
		constexpr				IEVector3(const float v[VectorW]);
								IEVector3(const IEVector4&);
								IEVector3(const IEVector3&) = default;
								~IEVector3() = default;

		// Constant Vectors
		static const IEVector3	XAxis;
		static const IEVector3	YAxis;
		static const IEVector3	ZAxis;
		static const IEVector3	ZeroVector;

		// Accessors
		float					getX() const;
		float					getY() const;
		float					getZ() const;
		const float*			getData() const;

		// Mutators
		void					setX(float);
		void					setY(float);
		void					setZ(float);
		void					setData(const float[VectorW]);
		IEVector3&				operator=(const IEVector3&) = default;
		float&					operator[](int);
		const float&			operator[](int) const;


		// Modify
		void					operator+=(const IEVector3&);
		void					operator-=(const IEVector3&);
		void					operator*=(const IEVector3&);
		void					operator*=(float);
		void					operator/=(const IEVector3&);
		void					operator/=(float);

		IEVector3				operator+(const IEVector3&) const;
		IEVector3				operator-(const IEVector3&) const;
		IEVector3				operator-() const;
		IEVector3				operator*(const IEVector3&) const;
		IEVector3				operator*(float) const;
		IEVector3				operator/(const IEVector3&) const;
		IEVector3				operator/(float) const;

		// Utilty
		float					DotProduct(const IEVector3&) const;
		IEVector3				CrossProduct(const IEVector3&) const;
		float					Length() const;
		float					LengthSqr() const;
		IEVector3				Normalize() const;
		IEVector3&				NormalizeSelf();
		IEVector3				Clamp(const IEVector3& min, const IEVector3& max) const;
		IEVector3				Clamp(float min, float max) const;
		IEVector3&				ClampSelf(const IEVector3& min, const IEVector3& max);
		IEVector3&				ClampSelf(float min, float max);

		// Logic
		bool					operator==(const IEVector3&) const;
		bool					operator!=(const IEVector3&) const;
};

// Requirements of Vector3
static_assert(std::is_literal_type<IEVector3>::value == true, "IEVector3 has to be literal type");
static_assert(std::is_trivially_copyable<IEVector3>::value == true, "IEVector3 has to be trivially copyable");
static_assert(std::is_polymorphic<IEVector3>::value == false, "IEVector3 should not be polymorphic");
static_assert(sizeof(IEVector3) == sizeof(float) * 3, "IEVector3 size is not 12 bytes");

// Left Scalar operators
IEVector3 operator*(float, const IEVector3&);

// Inlines
inline float IEVector3::getX() const {return x;}
inline float IEVector3::getY() const {return y;}
inline float IEVector3::getZ() const {return z;}
inline const float* IEVector3::getData() const {return v;}

inline void IEVector3::setX(float t) {x = t;}
inline void IEVector3::setY(float t) {y = t;}
inline void IEVector3::setZ(float t) {z = t;}
inline void IEVector3::setData(const float* t) {std::copy(t, t + VectorW, v);}

inline float& IEVector3::operator[](int index)
{
	assert(index >= 0 && index < VectorW);
	return v[index];
}

inline const float& IEVector3::operator[](int index) const
{
	assert(index >= 0 && index < VectorW);
	return v[index];
}

constexpr IEVector3::IEVector3()
	: x(0.0f)
	, y(0.0f)
	, z(0.0f)
{}

constexpr IEVector3::IEVector3(float xyz)
	: x(xyz)
	, y(xyz)
	, z(xyz)
{}

constexpr IEVector3::IEVector3(float xx, float yy, float zz)
	: x(xx)
	, y(yy)
	, z(zz)
{}

constexpr IEVector3::IEVector3(const float v[])
	: x(v[0])
	, y(v[1])
	, z(v[2])
{}