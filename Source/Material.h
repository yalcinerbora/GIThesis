/**

Material Definition

*/

#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <vector>
#include <cstdint>
#include "GLHeader.h"

struct ColorNormalMaterial
{
	char* colorFileName;
	char* normalFileName;
};

class Material
{
	private:
		std::vector<GLuint>				textures;
		GLuint							uniformBuffer;

		bool							dataEdited;
		std::vector<uint8_t>			uniformData;


	protected:
	public:
		// Constructors & Destructor
										Material(ColorNormalMaterial);
										Material(const Material&) = delete;
		const Material&					operator=(const Material&) = delete;
										~Material();

		// Equavilency Check
		bool							operator==(const Material&) const;
		bool							operator!=(const Material&) const;

		void							BindMaterial();
};
#endif //__MATERIAL_H__