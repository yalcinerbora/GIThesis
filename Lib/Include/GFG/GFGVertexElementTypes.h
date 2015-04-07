/**

GFGPosition Namespace
GFGNormal Namespace
GFGTangent Namespace
GFGBinormal Namespace
GFGWeight Namespace
GFGWeightIndex Namespace
GFGColor Namespace

	Predetermined format types of the Vertex
	Prob %80 of the computer graphics related apps only use these per vertex(atleast for raster graphics)
	Because of that these convinience classes supplied to make writing improters/exporters easier
	GFG definition does not classify vertex element types you can define your own vertex besides of these

	-- Static Mesh
	Position
	Normal
	Tangent
	Bitangent (Binormal)
	UV (Texture Coordinates)
	Color

	---Weighted Deformation
	Weight
	Weight Index


For License refer to:
https://github.com/yalcinerbora/GFGFileFormat/blob/master/LICENSE
*/

#ifndef __GFG_VERTEXELEMENTTYPES_H__
#define __GFG_VERTEXELEMENTTYPES_H__

#include "GFGEnumerations.h"

namespace GFGPosition
{
	bool		IsCompatible(GFGDataType);
	bool		ConvertData(uint8_t data[], size_t dataSize,
							const double pos[3], GFGDataType type);
	bool		UnConvertData(double pos[3], size_t dataSize,
							  const uint8_t data[], GFGDataType type);
};

namespace GFGNormal
{
	bool		IsCompatible(GFGDataType);
	bool		ConvertData(uint8_t data[], size_t dataSize,
							const double normal[3],
							GFGDataType type,
							const double tangent[3] = nullptr,
							const double bitangent[3] = nullptr);
	bool		UnConvertData(double normal[3], size_t dataSize,
							  const uint8_t data[],
							  GFGDataType type);
};

namespace GFGTangent
{
	bool		IsCompatible(GFGDataType);
	bool		ConvertData(uint8_t data[], size_t dataSize,
							const double tangent[3],
							GFGDataType type,
							const double normal[3] = nullptr,
							const double bitangent[3] = nullptr);
};

namespace GFGBinormal
{
	bool		IsCompatible(GFGDataType);
	bool		ConvertData(uint8_t data[], size_t dataSize,
							const double bitangent[3],
							GFGDataType type,
							const double tangent[3] = nullptr,
							const double normal[3] = nullptr);
};

namespace GFGUV
{
	bool		IsCompatible(GFGDataType);
	bool		ConvertData(uint8_t data[], size_t dataSize,
							const double uv[2],
							GFGDataType type);
	bool		UnConvertData(double uv[2], size_t dataSize,
							  const uint8_t data[], GFGDataType type);
};

namespace GFGWeight
{
	bool		IsCompatible(GFGDataType, unsigned int maxWeightInfluence);
	bool		ConvertData(uint8_t data[], size_t dataSize,
							const double weight[], unsigned int maxWeightInfluence,
							GFGDataType type);
	bool		UnConvertData(double weight[],
							  unsigned int &maxWeightInfluence,
							  size_t dataSize,
							  const uint8_t data[],
							  GFGDataType type);
};

namespace GFGWeightIndex
{
	bool		IsCompatible(GFGDataType, unsigned int maxWeightInfluence);
	bool		ConvertData(uint8_t data[], size_t dataSize,
							const unsigned int wIndex[], unsigned int maxWeightInfluence,
							GFGDataType type);
	bool		UnConvertData(unsigned int wIndex[],
							  unsigned int &maxWeightInfluence,
							  size_t dataSize,
							  const uint8_t data[],
							  GFGDataType type);
};

namespace GFGColor
{
	bool		IsCompatible(GFGDataType);
	bool		ConvertData(uint8_t data[], size_t dataSize,
							const double color[3],
							GFGDataType type);
	bool		UnConvertData(double color[3], size_t dataSize,
							  const uint8_t data[],
							  GFGDataType type);
};
#endif //__GFG_VERTEXELEMENTTYPES_H__