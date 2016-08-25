/**

Column Major Vector (4x1 Matrix) (NOT 1x4)

*/

#ifndef __IE_VECTOR4_H__
#define __IE_VECTOR4_H__

#include <algorithm>

class IEVector3;

class IEVector4
{
	private:
		union
		{
			struct				{float x, y, z, w;};
			float				v[4];
		};

	protected:

	public:
		// Constructors & Destructor
								IEVector4();
								IEVector4(float x, float y, float z, float w);
								IEVector4(const float v[4]);
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
		void					setData(const float[4]);
		IEVector4&				operator=(const IEVector4&) = default;

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

		// Logic
		bool					operator==(const IEVector4&) const;
		bool					operator!=(const IEVector4&) const;
};

// Requirements of Vector4
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
//inline IEVector4& IEVector4::operator=(const IEVector4& vector) { std::copy(vector.v, vector.v + 4, v); return *this; }

#endif //__IE_VECTOR4_H__