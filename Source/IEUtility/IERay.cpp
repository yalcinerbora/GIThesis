#ifndef USE_AVX

#include "IERay.h"
#include "IEMatrix3x3.h"
#include "IEMatrix4x4.h"
#include <cmath>

bool IERay::IntersectsSphere(IEVector3& intersectPos,
							 float& t,
							 const IEVector3& sphereCenter,
							 float sphereRadius) const
{
	// Geometric solution
	IEVector3 normDir = direction.Normalize();
	IEVector3 centerDir = sphereCenter - position;
	float beamCenterDistance = centerDir.DotProduct(normDir);
	float beamNormalLengthSqr = centerDir.DotProduct(centerDir) -
								beamCenterDistance * beamCenterDistance;
	float beamHalfLengthSqr = sphereRadius * sphereRadius - beamNormalLengthSqr;
	if(beamHalfLengthSqr > 0.0f)
	{
		// Inside Square
		float beamHalfLength = std::sqrt(beamHalfLengthSqr);
		float t0 = beamCenterDistance - beamHalfLength;
		float t1 = beamCenterDistance + beamHalfLength;
		if(t1 >= 0.0f)
		{
			t = (t0 >= 0.0f) ? t0 : t1;
			intersectPos = position + t * normDir;
			return true;
		}
	}
	return false;

	//// Quadratic Solution
	//const IEVector3 posDiff = position - sphereCenter;
	//float a = direction.DotProduct(direction);
	//float b = 2.0f * posDiff.DotProduct(direction);
	//float c = posDiff.DotProduct(posDiff) - sphereRadius * sphereRadius;
	//float delta = b * b - 4.0f * a * c;

	//float rayT0, rayT1;
	//if(delta > 0.0f)
	//{
	//	float q = (b > 0) ?
	//		-0.5 * (b + std::sqrt(delta)) :
	//		-0.5 * (b - std::sqrt(delta));
	//	rayT0 = q / a;

	//	// Mullers formula
	//	rayT1 = c / q;
	//}
	//else if(delta == 0.0f)
	//{
	//	rayT0 = rayT1 = -b * 0.5f / a;
	//}
	//else return false;

	//if(rayT0 < 0.0f && rayT1 < 0.0f) return false;
	//else if(rayT0 >= 0.0f && rayT1 >= 0.0f)
	//{
	//	t = std::min(rayT0, rayT1);
	//}
	//else if(rayT0 < 0.0f)
	//{
	//	t = rayT1;
	//}
	//else if(rayT1 < 0.0f)
	//{
	//	t = rayT0;
	//}
	//intersectPos = position + t * direction;
	//return true;
}

bool IERay::IntersectsTriangle(IEVector3& baryCoords,
							   float& t,
							   const IEVector3& t0,
							   const IEVector3& t1,
							   const IEVector3& t2) const
{
	// Matrix Solution
	// Kramers Rule
	auto abDiff = t0 - t1;
	auto acDiff = t0 - t2;
	auto aoDiff = t0 - position;

	IEMatrix3x3 A = IEMatrix3x3(abDiff, acDiff, direction);
	IEMatrix3x3 betaA = IEMatrix3x3(aoDiff, acDiff, direction);
	IEMatrix3x3 gammaA = IEMatrix3x3(abDiff, aoDiff, direction);
	IEMatrix3x3 tA = IEMatrix3x3(abDiff, acDiff, aoDiff);

	float aDetInv = 1.0f / A.Determinant();
	float beta = betaA.Determinant() * aDetInv;
	float gamma = gammaA.Determinant() * aDetInv;
	float alpha = 1.0f - beta - gamma;
	float rayT = tA.Determinant() * aDetInv;

	if(beta >= 0.0f && beta <= 1.0f &&
	   gamma >= 0.0f && gamma <= 1.0f &&
	   alpha >= 0.0f && alpha <= 1.0f &&
	   rayT >= 0.0f)
	{
		baryCoords = IEVector3(alpha, beta, gamma);
		t = rayT;
	}
	else return false;
	return true;
}

IERay IERay::Reflect(const IEVector3& normal) const
{
	float length = direction.Length();
	IEVector3 nDir = -direction / length;
	nDir = 2.0f * nDir.DotProduct(normal) * normal - nDir;
	return IERay(nDir * length, position);
}

IERay& IERay::ReflectSelf(const IEVector3& normal)
{
	float length = direction.Length();
	IEVector3 nDir = -direction / length;
	direction = (2.0f * nDir.DotProduct(normal) * normal - nDir) * length;
	return *this;
}

bool IERay::Refract(IERay& out, const IEVector3& normal, float fromMedium, float toMedium) const
{
	IEVector3 dir = direction.Normalize();
	float cosTetha = -normal.DotProduct(dir);
	float indexRatio = fromMedium / toMedium;

	float delta = 1.0f - indexRatio * indexRatio * (1.0f - cosTetha * cosTetha);
	if(delta > 0.0f)
	{
		out.direction = indexRatio * dir + normal * (cosTetha * indexRatio - std::sqrt(delta));
		out.position = position;
		return true;
	}
	return false;
}

bool IERay::RefractSelf(const IEVector3& normal, float fromMedium, float toMedium)
{
	IEVector3 dir = direction.Normalize();
	float cosTetha = -normal.DotProduct(dir);
	float indexRatio = fromMedium / toMedium;

	float delta = 1.0f - indexRatio * indexRatio * (1.0f - cosTetha * cosTetha);
	if(delta > 0.0f)
	{
		direction = indexRatio * dir + normal * (cosTetha * indexRatio - std::sqrt(delta));
		return true;
	}
	return false;
}


IERay IERay::NormalizeDir() const
{
	return IERay(direction.Normalize(), position);
}

IERay& IERay::NormalizeDirSelf()
{
	direction.NormalizeSelf();
	return *this;
}

IERay IERay::Advance(float t) const
{
	return IERay(direction, AdvancedPos(t));
}

IERay& IERay::AdvanceSelf(float t)
{
	position += t * direction;
	return *this;
}

IERay IERay::Transform(const IEMatrix4x4& mat) const
{
	return IERay(mat * IEVector4(direction, 0.0f),
				 mat * position);
}

IERay& IERay::TransformSelf(const IEMatrix4x4& mat)
{
	direction = mat * IEVector4(direction, 0.0f);
	position = mat * position;
	return *this;
}

IEVector3 IERay::AdvancedPos(float t) const
{
	return position + t * direction;
}

bool IERay::IntersectsAABB(const IEVector3& min, const IEVector3& max) const
{
	IEVector3 invD = IEVector3(1.0f, 1.0f, 1.0f) / direction;
	IEVector3 t0 = (min - position) * invD;
	IEVector3 t1 = (max - position) * invD;

	float tMin = -std::numeric_limits<float>::max();
	float tMax = std::numeric_limits<float>::max();

	for(int i = 0; i < 3; i++)
	{
		tMin = std::max(tMin, std::min(t0[i], t1[i]));
		tMax = std::min(tMax, std::max(t0[i], t1[i]));
	}

	return tMax >= tMin;
	//return (tMax - tMin) >= -1e-4;
}
#endif // USE_AVX