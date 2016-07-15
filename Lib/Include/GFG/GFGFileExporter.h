/**

GFGFileWriteI Interface
GFGFileWriterSTL Class
GFGAddMeshResult Struct
GFGFileExporter Class

File IO seperated with interface so you can use your faviourite
File API or you can use already implemented STL.

GFGFileExporter used to create new GFG Files.

For License refer to:
https://github.com/yalcinerbora/GFGFileFormat/blob/master/LICENSE
*/

#ifndef __GFG_FILEEXPORTER_H__
#define __GFG_FILEEXPORTER_H__

#include <vector>
#include <fstream>

#include "GFGEnumerations.h"
#include "GFGHeader.h"

// Seperation of File Reading and Layingout the file
class GFGFileWriterI
{
	private:
	protected:
	public:
		virtual void	Write(const uint8_t buffer[], size_t readAmount) = 0;
};

class GFGFileWriterSTL : public GFGFileWriterI
{
	private:
		std::ofstream&	writer;

	protected:
	public:
		// Constructors & Destructor
				GFGFileWriterSTL(std::ofstream& fileReader);

		void	Write(const uint8_t buffer[], size_t readAmount) override;
};

struct GFGAddMeshResult
{
	uint32_t meshIndex;
	uint32_t nodeIndex;
};

class GFGFileExporter
{
	private:
		// Properties
		// Header
		GFGHeader							gfgHeader;

		// Datas
		// Mesh Data
		std::vector<std::vector<uint8_t>>	meshData;
		std::vector<std::vector<uint8_t>>	meshIndexData;

		// Material Uniforms, Texture FilePaths
		std::vector<std::vector<uint8_t>>	materialUniformData;
		std::vector<std::vector<uint8_t>>	materialTexturePath;

		// Animation Data
		std::vector<std::vector<uint8_t>>	animationData;

	protected:
	public:
		// Constructors & Destructor
							GFGFileExporter() = default;
							~GFGFileExporter() = default;

		// Insertion
		GFGAddMeshResult	AddMesh(const GFGTransform& transform,
									uint32_t parent,
									const std::vector<GFGVertexComponent>& headerComponent,
									const GFGMeshHeaderCore& headerBase,
									const std::vector<uint8_t>& vertexData,
									const std::vector<uint8_t>* indexData = nullptr,
									const std::vector<GFGMeshMatPair>* materialPairings = nullptr,
									const std::vector<GFGMeshSkelPair>* skeletonPairings = nullptr);
		uint32_t			AddMesh(uint32_t parent,
									const std::vector<GFGVertexComponent>& headerComponent,
									const GFGMeshHeaderCore& headerBase,
									const std::vector<uint8_t>& vertexData,
									const std::vector<uint8_t>* indexData = nullptr,
									const std::vector<GFGMeshMatPair>* materialPairings = nullptr,
									const std::vector<GFGMeshSkelPair>* skeletonPairings = nullptr);
		uint32_t			AddSkeleton(const std::vector<uint32_t>& parentHierarcy,
										const std::vector<GFGTransform>& transforms);
		uint32_t			AddMaterial(GFGMaterialLogic logic,
										const std::vector<GFGTexturePath>* textureList = nullptr,
										const std::vector<GFGUniformData>* uniformList = nullptr,
										const std::vector<uint8_t>* texturePathData = nullptr,
										const std::vector<uint8_t>* uniformData = nullptr);
		uint32_t			AddNode(const GFGTransform& transform,
									uint32_t parent);
		uint32_t			AddAnimation(GFGAnimationLayout layout,
										 GFGAnimType type,
										 GFGQuatInterpType interpType,
										 GFGQuatLayout quatLayout,
										 uint32_t skeletonIndex,
										 uint32_t keyCount,
										 const std::vector<uint8_t>& animationData);

		void				Write(GFGFileWriterI&);
		void				Clear();

		// Access
		const GFGHeader&	Header();

};		

#endif //__GFG_FILEEXPORTER_H__