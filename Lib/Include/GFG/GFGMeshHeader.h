/**

GFGVertexComponent Structure
GFGMeshHeaderCore Structure
GFGMeshHeader Structure

Mesh Releated Structures used by GFGHeader class.

For License refer to:
https://github.com/yalcinerbora/GFGFileFormat/blob/master/LICENSE
*/

#ifndef __GFG_MESHHEADER_H__
#define __GFG_MESHHEADER_H__

#include <vector>
#include "GFGEnumerations.h"

// Pack Those Structs 
#pragma pack(push, 1)

// Components of this mesh
struct GFGVertexComponent
{
	GFGDataType				dataType;		// Type of the data
											// Size of this data type (in bytes) can be fetched using dataType
	GFGVertexComponentLogic logic;			// Component Logic(is this for position, tangent, normal, uv etc..)
	uint64_t				startOffset;	// Start location of this data structure relative to vertexStart (in bytes)
	uint64_t				internalOffset;	// Start location of this data relative to startOffset (in bytes)
	uint64_t				stride;			// Stride of this data structure (in bytes)
};

// Core Header
struct GFGMeshHeaderCore
{
	// Size Values
	uint64_t			vertexCount;	// # of vertices
	uint64_t			indexCount;		// # of indices
	uint32_t			indexSize;		// individual Index Size (in bytes)

	// Topology
	GFGTopology			topology;

	uint64_t			vertexStart;	// Starting Location of Vertex Data (byte offset relative to Data Start)
	uint64_t			indexStart;		// Starting Location of Index Data (byte offset relative to Data Start)

	// Components
	uint32_t			componentCount;	// # of components
};

struct GFGMeshHeader
{
	GFGMeshHeaderCore headerCore;
	std::vector<GFGVertexComponent> components;
};

static_assert (sizeof(GFGMeshHeaderCore) ==
			   sizeof(uint64_t) * 4
			   + sizeof(uint32_t) * 3,
			   "Mesh Header Core Size Mismatch from GFG Definition");

static_assert (sizeof(GFGVertexComponent) ==
			   sizeof(uint64_t) * 3
			   + sizeof(uint32_t) * 2,
			   "Vertex Component Size Mismatch from GFG Definition");

#pragma pack(pop)
#endif //__GFG_MESHHEADER_H__