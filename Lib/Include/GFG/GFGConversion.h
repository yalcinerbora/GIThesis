/**

GFGConversions Namespace

Used to convert various GFGDataType to their "meta" data type
for real numbers its "double" and for integers its "int"

For License refer to:
https://github.com/yalcinerbora/GFGFileFormat/blob/master/LICENSE
*/

#include "GFGEnumerations.h"

namespace GFGConversions
{
	// Float Conversions
	uint16_t			FloatToHalf(float);
	uint16_t			DoubleToHalf(double);

	uint8_t				DoubleToUnorm8(double);
	uint16_t			DoubleToUnorm16(double);
	uint32_t			DoubleToUnorm32(double);

	int8_t				DoubleToNorm8(double);
	int16_t				DoubleToNorm16(double);
	int32_t				DoubleToNorm32(double);
	
	uint8_t				FloatToUnorm8(float);
	uint16_t			FloatToUnorm16(float);
	uint32_t			FloatToUnorm32(float);

	int8_t				FloatToNorm8(float);
	int16_t				FloatToNorm16(float);
	int32_t				FloatToNorm32(float);

	// Special Multi Component Packing
	uint32_t			IntsToInt2_10_10_10(const int[4]);
	uint32_t			FloatsToInt2_10_10_10(const float[4]);
	uint32_t			DoublesToInt2_10_10_10(const double[4]);

	uint32_t			UIntsToUInt2_10_10_10(const unsigned int[4]);
	uint32_t			FloatsToUInt2_10_10_10(const float[4]);
	uint32_t			DoublesToUInt2_10_10_10(const double[4]);

	uint32_t			FloatsToUInt10F_11F_11F(const float[3]);
	uint32_t			DoublesToUInt10F_11F_11F(const double[3]);

	// Normal Packing
	uint32_t			FloatsToCustom_1_15N_16N(const float[3]);
	uint32_t			DoublesToCustom_1_15N_16N(const double[3]);
		
	// Tangent Packing
	void				FloatsToCustom_Tang_H_2N(uint32_t result[3],
												 const float normal[3],
												 const float tangent[3],
												 const float binormal[3]);
	void				DoublesToCustom_Tang_H_2N(uint32_t result[3],
												  const double normal[3],
												  const double tangent[3],
												  const double binormal[3]);

	// Multi Weight
	void				 DoubleToUnorm16_2_4(uint8_t dataOut[],
											 size_t dataCapacity,
											 const double data[],
											 unsigned int maxWeightInfluence);
	void				 DoubleToUnorm8_4_4(uint8_t dataOut[],
											size_t dataCapacity,
											const double data[],
											unsigned int maxWeightInfluence);
	void				 UIntToUInt16_2_4(uint8_t dataOut[],
										  size_t dataCapacity,
										  const unsigned int data[],
										  unsigned int maxWeightInfluence);
	void				 UIntToUInt8_4_4(uint8_t dataOut[],
										 size_t dataCapacity,
										 const unsigned int data[],
										 unsigned int maxWeightInfluence);

	// Multi Packing Generic
	void				DoubleToHalfV(uint8_t dataOut[], size_t dataCapacity, const double data[], size_t dataAmount);
	void				DoubleToFloatV(uint8_t dataOut[], size_t dataCapacity, const double data[], size_t dataAmount);
	void				DoubleToQuadV(uint8_t dataOut[], size_t dataCapacity, const double data[], size_t dataAmount);
	   
	void				DoubleToUnorm8V(uint8_t dataOut[], size_t dataCapacity, const double data[], size_t dataAmount);
	void				DoubleToUnorm16V(uint8_t dataOut[], size_t dataCapacity, const double data[], size_t dataAmount);
	void				DoubleToUnorm32V(uint8_t dataOut[], size_t dataCapacity, const double data[], size_t dataAmount);
	   
	void				DoubleToNorm8V(uint8_t dataOut[], size_t dataCapacity, const double data[], size_t dataAmount);
	void				DoubleToNorm16V(uint8_t dataOut[], size_t dataCapacity, const double data[], size_t dataAmount);
	void				DoubleToNorm32V(uint8_t dataOut[], size_t dataCapacity, const double data[], size_t dataAmount);
	   
	void				UIntToUInt8V(uint8_t dataOut[], size_t dataCapacity, const unsigned int data[], size_t dataAmount);
	void				UIntToUInt16V(uint8_t dataOut[], size_t dataCapacity, const unsigned int data[], size_t dataAmount);
	void				UIntToUInt32V(uint8_t dataOut[], size_t dataCapacity, const unsigned int data[], size_t dataAmount);
	   
	void				IntToInt8V(uint8_t dataOut[], size_t dataCapacity, const int data[], size_t dataAmount);
	void				IntToInt16V(uint8_t dataOut[], size_t dataCapacity, const int data[], size_t dataAmount);
	void				IntToInt32V(uint8_t dataOut[], size_t dataCapacity, const int data[], size_t dataAmount);

	// UNPACK //
	// Data Unpack (to Doubles)
	void				HalfToDoubleV(double dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);
	void				FloatToDoubleV(double dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);
	void				QuadToDoubleV(double dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);

	void				UNorm8ToDoubleV(double dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);
	void				UNorm16ToDoubleV(double dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);
	void				UNorm32ToDoubleV(double dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);

	void				Norm8ToDoubleV(double dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);
	void				Norm16ToDoubleV(double dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);
	void				Norm32ToDoubleV(double dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);

	// Data Unpack (to Generic Int)
	void				UInt8ToUIntV(unsigned  int dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);
	void				UInt16ToUIntV(unsigned int dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);
	void				UInt32ToUIntV(unsigned int dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);

	void				Int8ToIntV(int dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);
	void				Int16ToIntV(int dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);
	void				Int32ToIntV(int dataOut[], size_t dataCapacity, const uint8_t dataIn[], size_t dataAmount);

	// Special Multi Component UnPacking
	void				Int2_10_10_10ToInts(int dataOut[4], uint32_t data);
	void				Int2_10_10_10ToFloats(float dataOut[4], uint32_t data);
	void				Int2_10_10_10ToDoubles(double dataOut[4], uint32_t data);

	void				UInt2_10_10_10ToUInts(unsigned int dataOut[4], uint32_t data);
	void				UInt2_10_10_10ToFloats(float dataOut[4], uint32_t data);
	void				UInt2_10_10_10ToDoubles(double dataOut[4], uint32_t data);

	void				UInt10F_11F_11FToFloats(float dataOut[3], uint32_t data);
	void				UInt10F_11F_11FToDoubles(double dataOut[3], uint32_t data);

	// Normal UnPacking
	void				Custom_1_15N_16NToFloats(float dataOut[3], uint32_t data);
	void				Custom_1_15N_16NToDoubles(double dataOut[3], uint32_t data);

	// Multi Weight
	void				 Unorm16_2_4ToDoubles(double dataOut[],
											  unsigned int& maxWeightInfluence,
											  size_t dataCapacity,
											  const uint8_t data[]);
	void				 Unorm8_4_4ToDoubles(double dataOut[],
											 unsigned int& maxWeightInfluence,
											 size_t dataCapacity,
											 const uint8_t data[]);
	void				 UInt16_2_4ToUInts(unsigned int dataOut[],
										   unsigned int& maxWeightInfluence,
										   size_t dataCapacity,
										   const uint8_t data[]);
	void				 UInt8_4_4ToUInts(unsigned int dataOut[],
										  unsigned int& maxWeightInfluence,
										  size_t dataCapacity,
										  const uint8_t data[]);

	// TODO Add mode Unpacking Modes
};
