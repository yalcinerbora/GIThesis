#pragma once
/**

Column Major Vector (2x1 Matrix) (NOT 1x2)

*/

#include <algorithm>
#include <cassert>

class IEVector2
{
	private:
		static constexpr int	VectorW = 2;
		union
		{
			struct				{float x, y;};
			float				v[VectorW];
		};

	protected:

	public:
		// Constructors & Destructor
		constexpr				IEVector2();
		constexpr				IEVector2(float xy);
		constexpr				IEVector2(float x, float y);
		constexpr				IEVector2(const float v[VectorW]);
								IEVector2(const IEVector2&) = default;
								~IEVector2() = default;

		// Constant Vectors
		static const IEVector2	Xaxis;
		static const IEVector2	Yaxis;
		static const IEVector2	ZeroVector;

		// Accessors
		float					getX() const;
		float					getY() const;
		const float*			getData() const;

		// Mutators
		void					setX(float);
		void					setY(float);
		void					setData(const float[VectorW]);
		IEVector2&				operator=(const IEVector2&) = default;
		float&					operator[](int);
		const float&			operator[](int) const;


		// Modify
		void					operator+=(const IEVector2&);
		void					operator-=(const IEVector2&);
		void					operator*=(const IEVector2&);
		void					operator*=(float);
		void					operator/=(const IEVector2&);
		void					operator/=(float);

		IEVector2				operator+(const IEVector2&) const;
		IEVector2				operator-(const IEVector2&) const;
		IEVector2				operator-() const;
		IEVector2				operator*(const IEVector2&) const;
		IEVector2				operator*(float) const;
		IEVector2				operator/(const IEVector2&) const;
		IEVector2				operator/(float) const;

		// Utilty
		float					DotProduct(const IEVector2&) const;
		float					Length() const;
		float					LengthSqr() const;
		IEVector2				Normalize() const;
		IEVector2&				NormalizeSelf();
		IEVector2				Clamp(const IEVector2& min, const IEVector2& max) const;
		IEVector2				Clamp(float min, float max) const;
		IEVector2&				ClampSelf(const IEVector2& min, const IEVector2& max);
		IEVector2&				ClampSelf(float min, float max);

		// Logic
		bool					operator==(const IEVector2&) const;
		bool					operator!=(const IEVector2&) const;
};

// Requirements of Vector3
static_assert(std::is_literal_type<IEVector2>::value == true, "IEVector2 has to be literal type");
static_assert(std::is_trivially_copyable<IEVector2>::value == true, "IEVector2 has to be trivially copyable");
static_assert(std::is_polymorphic<IEVector2>::value == false, "IEVector2 should not be polymorphic");
static_assert(sizeof(IEVector2) == sizeof(float) * 2, "IEVector2 size is not 8 bytes");

// Left Scalar operators
IEVector2 operator*(float, const IEVector2&);

// Inlines
inline float IEVector2::getX() const {return x;}
inline float IEVector2::getY() const {return y;}
inline const float* IEVector2::getData() const {return v;}

inline void IEVector2::setX(float t) {x = t;}
inline void IEVector2::setY(float t) {y = t;}
inline void IEVector2::setData(const float* t) {std::copy(t, t + VectorW, v);}

inline float& IEVector2::operator[](int index)
{
	assert(index >= 0 && index < VectorW);
	return v[index];
}

inline const float& IEVector2::operator[](int index) const
{
	assert(index >= 0 && index < VectorW);
	return v[index];
}

constexpr IEVector2::IEVector2()
	: x(0.0f)
	, y(0.0f)
{}

constexpr IEVector2::IEVector2(float xy)
	: x(xy)
	, y(xy)
{}

constexpr IEVector2::IEVector2(float xx, float yy)
	: x(xx)
	, y(yy)
{}

constexpr IEVector2::IEVector2(const float v[])
	: x(v[0])
	, y(v[1])
{}