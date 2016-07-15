/**

GFGTransform Strcuture
GFGTransformList Structure
GFGNode Strcuture
GFGHierarchy Structure
GFGMeshJumpList Structure
GFGAnimationJumpList Structure
GFGMaterialHeader Structure
GFGSkeletonJumpList Structure
GFGMeshMatPair Structure
GFGMeshSkelPair Structure
GFGMeshSkelPairList Structure
GFGHeader Class

GFGHeader class hold the variable sized GFGHeader "serializes" data for file write

Various enumerations used by the GFGHeader.

For License refer to:
https://github.com/yalcinerbora/GFGFileFormat/blob/master/LICENSE
*/

#ifndef __GFG_HEADER_H__
#define __GFG_HEADER_H__

#include "GFGEnumerations.h"
#include "GFGMeshHeader.h"
#include "GFGMaterialHeader.h"
#include "GFGSkeletonHeader.h"
#include "GFGAnimationHeader.h"

// FourCC Code
static const uint32_t GFGFourCC = ' ' << 24 |
								  'G' << 16 | 
								  'F' << 8 | 
								  'G' << 0;

struct GFGTransform
{
	// World(Model) Transform definiton
	// Pivot point is (0, 0, 0) of the local space
	// Euler Ordering X, Y, Z,
	// Transform order Scale then rotate then translate
	float		translate[3];		// Translation
	float		rotate[3];			// Rotation (eul) (x, y, z axis)
	float		scale[3];			// Scale
};

struct GFGTransformList
{
	uint32_t					transformAmount;
	std::vector<GFGTransform>	transforms;
};

struct GFGNode
{
	uint32_t	parentIndex;		// This index is for this array
	uint32_t	transformIndex;		// Transform Data
	uint32_t	meshReference;		// Index of the mesh mat
};

struct GFGHierarchy
{
	uint32_t				nodeAmount;
	std::vector<GFGNode>	nodes;
};

struct GFGMeshJumpList
{
	uint32_t				nodeAmount;
	std::vector<uint64_t>	meshLocations;
};

struct GFGAnimationJumpList
{
	uint32_t				nodeAmount;
	std::vector<uint64_t>	animationLocations;
};

struct GFGMaterialJumpList
{
	uint32_t				nodeAmount;
	std::vector<uint64_t>	materialLocations;
};

struct GFGSkeletonJumpList
{
	uint32_t				nodeAmount;
	std::vector<uint64_t>	skeletonLocations;
};

// Mesh Material Connections
// Multiple Mesh - Multiple Mat
struct GFGMeshMatPair
{
	uint32_t	meshIndex;
	uint32_t	materialIndex;
	uint64_t	indexOffset;	// Sub portion of the mesh which will be rendered with this mat
	uint64_t	indexCount;		// Index Count 
								// If mesh is not indexed these shows vertex offsets
								// and vertex counts
};

struct GFGMeshMatPairList
{
	uint32_t					meshMatCount;
	std::vector<GFGMeshMatPair> pairs;
};

// Mesh Skeleton Connections
// Multi Mesh - Multi Skeleton
struct GFGMeshSkelPair
{
	uint32_t	meshIndex;
	uint32_t	skeletonIndex;
};

struct GFGMeshSkeletonPairList
{
	uint32_t						meshSkelCount;
	std::vector<GFGMeshSkelPair>	connections;
};

// Header Block
// Variable
class GFGHeader
{
	private:
	protected:
	public:
		// This also shows order of the header
		uint32_t						fourCC = GFGFourCC;	// From MSB to LSB " GFG"
		uint64_t						headerSize;			// (in bytes) Header Size also show the data start location
		uint64_t						transformJump;		// Direct Jump to transform headers (convinience)

		GFGMeshJumpList					meshList;			// Fast Jumping to the headers (since headers are variable sized)
		GFGMaterialJumpList				materialList;		// Fast Jumping to the headers (since headers are variable sized)
		GFGSkeletonJumpList				skeletonList;		// Fast Jumping to the headers (since headers are variable sized)
		GFGAnimationJumpList			animationList;		// Future Compat always zero atm
	
		GFGHierarchy					sceneHierarchy;		// Scene Hierarchy

		// Data Connections
		GFGMeshMatPairList				meshMaterialConnections;
		GFGMeshSkeletonPairList			meshSkeletonConnections;
	
		// Data References
		std::vector<GFGMeshHeader>		meshes;
		std::vector<GFGMaterialHeader>	materials;
		std::vector<GFGSkeletonHeader>	skeletons;
		std::vector<GFGAnimationHeader>	animations;

		// Transform
		GFGTransformList				transformData;
		GFGTransformList				bonetransformData;			// This should be "bind pose"

		// Utility
		void							CalculateDataOffsets();
		void							Clear();
};
#endif //__GFG_HEADER_H__