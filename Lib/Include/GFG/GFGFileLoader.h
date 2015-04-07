/**

GFGFileReaderI Interface
GFGFileReaderSTL Class
GFGFileLoader Class
GFGFileError Enumeration


GFGFileReaderI is required interface for GFGFileLoader class.
GFGGFileReaderI provides required file io functions needed by the
file loader. You can use your favourice file handling library or
you can used the already implemented STL one.

GFGFileLoader is used to fetch data from GFG File.

GFGFileError Enumerations holds errors can happen during validation,
or data fetch operations.

For License refer to:
https://github.com/yalcinerbora/GFGFileFormat/blob/master/LICENSE
*/

#ifndef __GFG_FILELOADER_H__
#define __GFG_FILELOADER_H__

#include "GFGHeader.h"
#include "GFGEnumerations.h"
#include <fstream>

// TODO: maybe user does not want to include <fstream>
// Seperation of File Reading and Layout of the file
class GFGFileReaderI
{
	private:
	protected:
	public:
		virtual void	Read(uint8_t buffer[], size_t readAmount) = 0;
		virtual void	MovePtrAbs(size_t absLocation) = 0;
		virtual void	MovePtrRelative(int64_t relLocation, GFGDirection) = 0;
		virtual size_t	GetFileSize() = 0;
};

class GFGFileReaderSTL : public GFGFileReaderI
{
	private:
		std::ifstream&	reader;

	protected:
	public:
		// Constructors & Destructor
				GFGFileReaderSTL(std::ifstream& fileReader);

		void	Read(uint8_t buffer[], size_t readAmount) override;
		void	MovePtrAbs(size_t absLocation)  override;
		void	MovePtrRelative(int64_t relLocation, GFGDirection dir)  override;
		size_t	GetFileSize() override;
};

// GFG File Errors
enum class GFGFileError
{
	OK,
	FILE_CANNOT_CONTAIN_HEADER, // HeaderSize > FileSize
	DATA_OFFSET_WRONG,			// Absolute data offset > FileSize
	FILE_FOURCC_MISMATCH		// FourCC code is not 'GFG '
};

class GFGFileLoader
{
	private:
		static size_t					EmptyHeaderSize;

		// Properties
		GFGHeader						header;
		GFGFileReaderI*					reader;
		bool							valid;

	protected:

	public:
		// Constructors & Destructor
										GFGFileLoader();
										GFGFileLoader(GFGFileReaderI* reader);
		const GFGFileLoader&			operator=(const GFGFileLoader&) = delete;
		GFGFileLoader&					operator=(GFGFileLoader&&);
										GFGFileLoader(const GFGFileLoader&) = delete;
										~GFGFileLoader() = default;

		// Header Access
		const GFGHeader&				Header() const;

		// Exporting
		GFGFileError					ValidateAndOpen();

		// Data Segment Export
		// Mesh Importing
		GFGFileError					MeshVertexData(uint8_t data[], uint32_t meshIndex);
		GFGFileError					AllMeshVertexData(uint8_t data[]);
		GFGFileError					MeshIndexData(uint8_t data[], uint32_t meshIndex);
		GFGFileError					AllMeshIndexData(uint8_t data[]);

		// Material Importing
		GFGFileError					MaterialTextureData(uint8_t data[], uint32_t materialIndex);
		GFGFileError					AllMaterialTextureData(uint8_t data[]);
		GFGFileError					MaterialUniformData(uint8_t data[], uint32_t materialIndex);
		GFGFileError					AllMaterialUniformData(uint8_t data[]);

		// Skeleton Importing
		// Skeleton Does not have data segment

		// Animation Importing
		GFGFileError					AnimationKeyframeData(GFGTransform data[], uint32_t animIndex);
		GFGFileError					AllAnimationKeyframeData(GFGTransform data[]);

		// Data Byte Sizes
		uint64_t						MeshVertexDataSize(uint32_t meshIndex) const;
		uint64_t						AllMeshVertexDataSize() const;
		uint64_t						MeshIndexDataSize(uint32_t meshIndex) const;
		uint64_t						AllMeshIndexDataSize() const;

		uint64_t						MaterialTextureDataSize(uint32_t materialIndex) const;
		uint64_t						AllMaterialTextureDataSize() const;
		uint64_t						MaterialUniformDataSize(uint32_t materialIndex) const;
		uint64_t						AllMaterialUniformDataSize() const;

		uint64_t						AnimationKeyframeDataSize(uint32_t animIndex) const;
		uint64_t						AllAnimationKeyframeDataSize()const;


};
#endif //__GFG_FILELOADER_H__