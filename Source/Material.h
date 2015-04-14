/**

Material Definition

*/

#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <vector>
#include <cstdint>
#include "GLHeader.h"

struct BasicPhong
{

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
										Material(BasicPhong);
										Material(const Material&) = delete;
		const Material&					operator=(const Material&) = delete;
										~Material();

		// 
		void							BindMaterial();

};
#endif //__MATERIAL_H__