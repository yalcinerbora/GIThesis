/**
Random Number Generator c++11 Implementation

Little Bit More User Friendly

Author(s):
Bora Yalciner
*/
#ifndef __IE_RANDOM_H__
#define __IE_RANDOM_H__

#include <random>
#include "IETypes.h"

class IEVector3;
class IEVector4;
class IEQuaternion;
class IEMatrix4x4;

class IERandom
{
	private:
		std::mt19937_64			engine;

	protected:

	public:
		// Constructors & Destructor
								IERandom();					// Random Seed
								IERandom(IEUInt64 seed);	// Your Seed
								~IERandom() = default;

		// Integer Types
		IEUInt64				IntMM(IEUInt64 min, IEUInt64 max);
		IEUInt64				IntMV(IEUInt64 mean, IEUInt64 variance);
		IEUInt64				Int(IEUInt64 max);
		IEUInt64				Int();

		// Floating point
		double					Double01();
		double					Double(double mean, double variance);

		// Algebraic Types
		void					Vec3(IEVector3& result,
									 const IEVector3& mean,
									 const IEVector3& variance);
		void					Vec4(IEVector4& result,
									 const IEVector4& mean,
									 const IEVector4& variance);
		void					Quat(IEQuaternion& result,
									 const IEQuaternion& mean,
									 const IEQuaternion& variance);
		void					Mat4x4(IEMatrix4x4& result,
									   const IEMatrix4x4& mean,
									   const IEMatrix4x4& variance);

		// Generic Array Types
		void					IntArrayMM(IEUInt64 result[],
										   const IEUInt64 min[],
										   const IEUInt64 max[],
										   IESize size);
		void					IntArrayMV(IEUInt64 result[],
										   const IEUInt64 mean[],
										   const IEUInt64 variance[],
										   IESize size);
		void					DoubleArray01(double result[], IESize size);
		void					DoubleArray(double result[],
											const double mean[],
											const double variance[],
											IESize size);
};
#endif //__IE_RANDOM_H__