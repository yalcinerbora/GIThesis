#pragma once
/**

*/

class IEVector3;
class IEVector4;
class IEQuaternion;
class IEMatrix4x4;
class IEMatrix3x3;

namespace IEFunctions
{
	template <class T>
	T Clamp(const T& x, const T& min, const T& max)
	{
		return (x < min) ? min : ((x > max) ? max : x);
	}

	template <class T>
	T Lerp(const T& from, const T& to, float alpha)
	{
		assert(alpha >= 0.0f && alpha <= 1.0f);
		return to * alpha + from * (1.0f - alpha);
	}
	
	template <class T>
	T DefaultInterper(const T& from, const T& to, float alpha)
	{
		return Lerp<T>(from, to, alpha);
	}

	template<>
	IEVector3 Clamp(const IEVector3& x, const IEVector3& min, const IEVector3& max);

	template<>
	IEVector4 Clamp(const IEVector4& x, const IEVector4& min, const IEVector4& max);

	template<>
	IEMatrix3x3 Clamp(const IEMatrix3x3& x, const IEMatrix3x3& min, const IEMatrix3x3& max);

	template<>
	IEMatrix4x4 Clamp(const IEMatrix4x4& x, const IEMatrix4x4& min, const IEMatrix4x4& max);

	template<>
	IEVector3 Lerp(const IEVector3& from, const IEVector3& to, float alpha);

	template<>
	IEVector4 Lerp(const IEVector4& from, const IEVector4& to, float alpha);

	template<>
	IEQuaternion Lerp(const IEQuaternion& from, const IEQuaternion& to, float alpha);

	template<>
	IEMatrix3x3 Lerp(const IEMatrix3x3& from, const IEMatrix3x3& to, float alpha);

	template<>
	IEMatrix4x4 Lerp(const IEMatrix4x4& from, const IEMatrix4x4& to, float alpha);
};