/**

Material Definition

*/

#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <string>
#include <vector>
#include <cstdint>
#include "GLHeader.h"
#include "IEUtility/IEVector3.h"

struct ColorMaterial
{
	std::string		colorFileName;
	std::string		normalFileName;
};

class Material
{
	private:
		static constexpr GLsizei		levels = 4;
		GLuint							texture;
		GLuint							sampler;

	protected:
	public:
		// Constructors & Destructor
										Material(ColorMaterial);
										Material(Material&&);
										Material(const Material&) = delete;
		Material&						operator=(const Material&) = delete;
										~Material();

		// Equavilency Check
		void							BindMaterial();
};
#endif //__MATERIAL_H__