#include "IERandom.h"
#include "IEVector3.h"
#include "IEVector4.h"
#include "IEMatrix4x4.h"
#include "IEQuaternion.h"

IERandom::IERandom()
{
	std::random_device rd;
	engine.seed(rd());
}

IERandom::IERandom(uint64_t seed) : engine(seed)
{}


uint64_t IERandom::IntMM(uint64_t min, uint64_t max)
{
	assert(min < max);
	assert(max <= std::mt19937_64::max());
	assert(min >= std::mt19937_64::min());
	return static_cast<uint64_t>(((static_cast<double>(engine()) / std::mt19937_64::max()) * (max - min))) + min;
}

uint64_t IERandom::IntMV(uint64_t mean, uint64_t variance)
{
	assert(mean + variance <= std::mt19937_64::max());
	assert(mean - variance >= std::mt19937_64::min());
	return static_cast<uint64_t>(((static_cast<double>(engine()) / std::mt19937_64::max()) * 2 * variance)) + mean - variance;
}

uint64_t IERandom::Int(uint64_t max)
{
	assert(max <= std::mt19937_64::max());
	return static_cast<uint64_t>(((static_cast<double>(engine()) / std::mt19937_64::max()) * max));
}

uint64_t IERandom::Int()
{
	return engine();
}

double IERandom::Double01()
{
	return (static_cast<double>(engine()) / std::mt19937_64::max());
}

double IERandom::Double(double mean, double variance)
{
	assert(variance >= 0);
	return ((static_cast<double>(engine()) / std::mt19937_64::max()) * 2 * variance) + mean - variance;
}

void IERandom::Vec3(IEVector3& result,
					const IEVector3& mean,
					const IEVector3& variance)
{
	result.setX(static_cast<float>(Double(mean.getX(), variance.getX())));
	result.setY(static_cast<float>(Double(mean.getY(), variance.getY())));
	result.setZ(static_cast<float>(Double(mean.getZ(), variance.getZ())));
}

void IERandom::Vec4(IEVector4& result,
					const IEVector4& mean,
					const IEVector4& variance)
{
	result.setX(static_cast<float>(Double(mean.getX(), variance.getX())));
	result.setY(static_cast<float>(Double(mean.getY(), variance.getY())));
	result.setZ(static_cast<float>(Double(mean.getZ(), variance.getZ())));
	result.setW(static_cast<float>(Double(mean.getW(), variance.getW())));
}

void IERandom::Quat(IEQuaternion& result,
					const IEQuaternion& mean,
					const IEQuaternion& variance)
{
	result.setX(static_cast<float>(Double(mean.getX(), variance.getX())));
	result.setY(static_cast<float>(Double(mean.getY(), variance.getY())));
	result.setZ(static_cast<float>(Double(mean.getZ(), variance.getZ())));
	result.setW(static_cast<float>(Double(mean.getW(), variance.getW())));
}

void IERandom::Mat4x4(IEMatrix4x4& result,
					  const IEMatrix4x4& mean,
					  const IEMatrix4x4& variance)
{
	for(int i = 0; i < 16; i++)
	{
		result.setElement(i / 4,
						  i % 4,
						  static_cast<float>(Double(mean(i / 4, i % 4), variance(i / 4, i % 4))));
	}
}

void IERandom::IntArrayMM(uint64_t result[],
						  const uint64_t min[],
						  const uint64_t max[],
						  size_t size)
{
	for(unsigned int i = 0; i < size; i++)
	{
		result[i] = IntMM(min[i], max[i]);
	}
}

void IERandom::IntArrayMV(uint64_t result[],
						   const uint64_t mean[],
						   const uint64_t variance[],
						   size_t size)
{
	for(unsigned int i = 0; i < size; i++)
	{
		result[i] = IntMV(mean[i], variance[i]);
	}
}
void IERandom::DoubleArray01(double result[], size_t size)
{
	for(unsigned int i = 0; i < size; i++)
	{
		result[i] = Double01();
	}
}
void IERandom::DoubleArray(double result[],
							const double mean[],
							const double variance[],
							size_t size)
{
	for(unsigned int i = 0; i < size; i++)
	{
		result[i] = Double(mean[i], variance[i]);
	}
}