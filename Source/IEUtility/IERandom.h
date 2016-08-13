/**
Random Number Generator c++11 Implementation

Little Bit More User Friendly

Author(s):
Bora Yalciner
*/
#ifndef __IE_RANDOM_H__
#define __IE_RANDOM_H__

#include <random>
#include <cstdint>

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
								IERandom(uint64_t seed);	// Your Seed
								~IERandom() = default;

		// Integer Types
		uint64_t				IntMM(uint64_t min, uint64_t max);
		uint64_t				IntMV(uint64_t mean, uint64_t variance);
		uint64_t				Int(uint64_t max);
		uint64_t				Int();

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
		void					IntArrayMM(uint64_t result[],
										   const uint64_t min[],
										   const uint64_t max[],
										   size_t size);
		void					IntArrayMV(uint64_t result[],
										   const uint64_t mean[],
										   const uint64_t variance[],
										   size_t size);
		void					DoubleArray01(double result[], size_t size);
		void					DoubleArray(double result[],
											const double mean[],
											const double variance[],
											size_t size);
};
#endif //__IE_RANDOM_H__