#pragma once
/**

Column Major Vector (4x1 Matrix) (NOT 1x4)

*/
#include <algorithm>
#include <immintrin.h>

class IEVector3;
class IEVector4
{
	private:
		static constexpr int	VectorW = 4;
		union
		{
			__m128				avx;
			struct				{float x, y, z, w;};
			float				v[VectorW];
		};

	protected:

	public:
		// Constructors & Destructor
		constexpr				IEVector4();
		constexpr				IEVector4(float xyzw);
		constexpr				IEVector4(float x, float y, float z, float w);
		constexpr				IEVector4(const float v[VectorW]);
								IEVector4(const IEVector3&);
								IEVector4(const IEVector3&, float w);
								//IEVector4(const IEVector4&) = default;
								//~IEVector4() = default;

		// Statics
		static const IEVector4	ZeroVector;
		static const IEVector4	XAxis;
		static const IEVector4	YAxis;
		static const IEVector4	ZAxis;

		// Accessors
		float					getX() const;
		float					getY() const;
		float					getZ() const;
		float					getW() const;
		const float*			getData() const;

		// Mutators
		void					setX(float);
		void					setY(float);
		void					setZ(float);
		void					setW(float);
		void					setData(const float[VectorW]);
		IEVector4&				operator=(const IEVector4&) = default;
		float&					operator[](int);
		const float&			operator[](int) const;

		// Modify
		void					operator+=(const IEVector4&);
		void					operator-=(const IEVector4&);
		void					operator*=(const IEVector4&);
		void					operator*=(float);
		void					operator/=(const IEVector4&);
		void					operator/=(float);

		IEVector4				operator+(const IEVector4&) const;
		IEVector4				operator-(const IEVector4&) const;
		IEVector4				operator-() const;
		IEVector4				operator*(const IEVector4&) const;
		IEVector4				operator*(float) const;
		IEVector4				operator/(const IEVector4&) const;
		IEVector4				operator/(float) const;

		// Utilty
		float					DotProduct(const IEVector4&) const;
		float					Length() const;
		float					LengthSqr() const;
		IEVector4				Normalize() const;
		IEVector4&				NormalizeSelf();
		IEVector4				Clamp(const IEVector4&, const IEVector4&) const;
		IEVector4				Clamp(float min, float max) const;
		IEVector4&				ClampSelf(const IEVector4& min, const IEVector4& max);
		IEVector4&				ClampSelf(float min, float max);

		// Logic
		bool					operator==(const IEVector4&) const;
		bool					operator!=(const IEVector4&) const;
};

// Requirements of Vector4
static_assert(std::is_literal_type<IEVector4>::value == true, "IEVector4 has to be literal type");
static_assert(std::is_trivially_copyable<IEVector4>::value == true, "IEVector4 has to be trivially copyable");
static_assert(std::is_polymorphic<IEVector4>::value == false, "IEVector4 should not be polymorphic");
static_assert(sizeof(IEVector4) == sizeof(float) * 4, "IEVector4 size is not 16 bytes");

// Left Scalar operators
IEVector4 operator*(float, const IEVector4&);

// Inlines
inline float IEVector4::getX() const {return x;}
inline float IEVector4::getY() const { return y; }
inline float IEVector4::getZ() const { return z; }
inline float IEVector4::getW() const { return w; }
inline const float* IEVector4::getData() const { return v; }

inline void IEVector4::setX(float t) { x = t; }
inline void IEVector4::setY(float t) { y = t; }
inline void IEVector4::setZ(float t) { z = t; }
inline void IEVector4::setW(float t) { w = t; }
inline void IEVector4::setData(const float data[]) { std::copy(data, data + 4, v); }

inline float& IEVector4::operator[](int index)
{
	assert(index >= 0 && index < VectorW);
	return v[index];
}

inline const float& IEVector4::operator[](int index) const
{
	assert(index >= 0 && index < VectorW);
	return v[index];
}

constexpr IEVector4::IEVector4()
	: x(0.0f)
	, y(0.0f)
	, z(0.0f)
	, w(0.0f)
{}

constexpr IEVector4::IEVector4(float xyzw)
	: x(xyzw)
	, y(xyzw)
	, z(xyzw)
	, w(xyzw)
{}

constexpr IEVector4::IEVector4(float xx, float yy, float zz, float ww)
	: x(xx)
	, y(yy)
	, z(zz)
	, w(ww)
{}

constexpr IEVector4::IEVector4(const float v[])
	: x(v[0])
	, y(v[1])
	, z(v[2])
	, w(v[3])
{}