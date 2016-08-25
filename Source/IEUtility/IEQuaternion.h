/**

Column Major Vector Matrix

*/

#ifndef __IE_QUATERNION_H__
#define __IE_QUATERNION_H__

#define SLERP_TO_LERP_SWITCH_THRESHOLD 0.05f

#include <algorithm>
#include <cassert>

class IEVector3;

class IEQuaternion
{
	private:
		union
		{
			struct					{float w, x, y, z;};
			float					v[4];
		};

	protected:
		
	public:
		// Constructors & Destructor
									IEQuaternion();
									IEQuaternion(float w, float x, float y, float z);
									IEQuaternion(const float*);
									IEQuaternion(float angle, const IEVector3& axis);
									IEQuaternion(const IEQuaternion&) = default;
									~IEQuaternion() = default;
		
		// Constants
		static const IEQuaternion	IdentityQuat;
	
		// Accessors
		float						getW() const;
		float						getX() const;
		float						getY() const;
		float						getZ() const;
		const float*				getData() const;

		// Mutators
		void						setW(float);
		void						setX(float);
		void						setY(float);
		void						setZ(float);
		void						setData(const float[4]);
		IEQuaternion&				operator=(const IEQuaternion&) = default;

		// Operators
		IEQuaternion				operator*(const IEQuaternion&) const;
		IEQuaternion				operator*(float) const;
		IEQuaternion				operator+(const IEQuaternion&) const;
		IEQuaternion				operator-(const IEQuaternion&) const;
		IEQuaternion				operator-() const;
		IEQuaternion				operator/(float) const;

		void						operator*=(const IEQuaternion&);
		void						operator*=(float);
		void						operator+=(const IEQuaternion&);
		void						operator-=(const IEQuaternion&);
		void						operator/=(float);

		// Logic
		bool						operator==(const IEQuaternion&);
		bool						operator!=(const IEQuaternion&);

		// Utility
		IEQuaternion				Normalize() const;
		IEQuaternion&				NormalizeSelf();
		float						Length() const;
		float						LengthSqr() const;
		IEQuaternion				Conjugate() const;
		IEQuaternion&				ConjugateSelf();
		float						DotProduct(const IEQuaternion&) const;
		IEVector3					ApplyRotation(const IEVector3&);

		// Static Utility
		static IEQuaternion			NLerp(const IEQuaternion& start, const IEQuaternion& end, float percent);
		static IEQuaternion			SLerp(const IEQuaternion& start, const IEQuaternion& end, float percent);
};

// Requirements of IEQuaternion
static_assert(std::is_trivially_copyable<IEQuaternion>::value == true, "IEQuaternion has to be trivially copyable");
static_assert(std::is_polymorphic<IEQuaternion>::value == false, "IEQuaternion should not be polymorphic");
static_assert(sizeof(IEQuaternion) == sizeof(float) * 4, "IEQuaternion size is not 16 bytes");

// Left Scalar operators
IEQuaternion operator*(float, const IEQuaternion&);

// Inlines
inline float IEQuaternion::getW() const { return w; }
inline float IEQuaternion::getX() const { return x; }
inline float IEQuaternion::getY() const { return y; }
inline float IEQuaternion::getZ() const { return z; }
inline const float* IEQuaternion::getData() const { return v; }

// Mutators
inline void IEQuaternion::setW(float ww) { w = ww; }
inline void IEQuaternion::setX(float xx) { x = xx; }
inline void IEQuaternion::setY(float yy) { y = yy; }
inline void IEQuaternion::setZ(float zz) { z = zz; }
inline void IEQuaternion::setData(const float data[]) { std::copy(data, data + 4, v); }
//inline IEQuaternion& IEQuaternion::operator=(const IEQuaternion& quaternion) { std::copy(quaternion.v, quaternion.v + 4, v); return *this; }

#endif //__IE_QUATERNION_H__