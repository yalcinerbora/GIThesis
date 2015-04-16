/**

Column Major Vector (3x1 Matrix) (NOT 1x3)

*/

#ifndef __IE_VECTOR3_H__
#define __IE_VECTOR3_H__

#include <algorithm>

class IEVector4;

class IEVector3
{
	private:
		union
		{
			struct				{float x, y, z;};
			float				v[3];
		};

	protected:

	public:
		// Constructors & Destructor
								IEVector3();
								IEVector3(float x, float y, float z);
								IEVector3(const float v[3]);
								IEVector3(const IEVector4&);
								IEVector3(const IEVector3&) = default;
								~IEVector3() = default;

		// Constant Vectors
		static const IEVector3	Xaxis;
		static const IEVector3	Yaxis;
		static const IEVector3	Zaxis;
		static const IEVector3	ZeroVector;

		// Accessors
		inline float			getX() const;
		inline float			getY() const;
		inline float			getZ() const;
		inline const float*		getData() const;

		// Mutators
		inline void				setX(float);
		inline void				setY(float);
		inline void				setZ(float);
		inline void				setData(const float[3]);
		inline IEVector3&		operator=(const IEVector3&);

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

		// Logic
		bool					operator==(const IEVector3&) const;
		bool					operator!=(const IEVector3&) const;
};

// Requirements of Vector3
static_assert(std::is_trivially_copyable<IEVector3>::value == true, "IEVector3 has to be trivially copyable");
static_assert(std::is_polymorphic<IEVector3>::value == false, "IEVector3 should not be polymorphic");
static_assert(sizeof(IEVector3) == sizeof(float) * 3, "IEVector3 size is not 12 bytes");

// Left Scalar operators
IEVector3 operator*(float, const IEVector3&);

// Inlines
float IEVector3::getX() const {return x;}
float IEVector3::getY() const {return y;}
float IEVector3::getZ() const {return z;}
const float* IEVector3::getData() const {return v;}

void IEVector3::setX(float t) {x = t;}
void IEVector3::setY(float t) {y = t;}
void IEVector3::setZ(float t) {z = t;}
void IEVector3::setData(const float* t) {std::copy(t, t + 3, v);}
IEVector3& IEVector3::operator=(const IEVector3& vector){std::copy(vector.v, vector.v + 3, v); return *this;}

#endif //__IE_VECTOR3_H__