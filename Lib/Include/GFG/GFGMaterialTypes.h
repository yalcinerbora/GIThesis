/**

GFGMayaLambertLoc Enumeration
GFGMayaPhongLoc Enumeration
GFGMayaDX11UniformLoc Enumeration
GFGMayaDX11TextureLoc Enumeration

These Enumerations holds what Texture/Uniform at that index for each material
holds.

(i.e GFGMaterialLogic::MAYA_PHONG holds diffuse factor at its 4th index)

For License refer to:
https://github.com/yalcinerbora/GFGFileFormat/blob/master/LICENSE
*/

#ifndef __GFG_MATERIALTYPES_H__
#define __GFG_MATERIALTYPES_H__

// Texture Index are one to one with Uniform Index
enum class GFGMayaLambertLoc
{
	COLOR,
	TRANSPARENCY,
	AMBIENT_COLOR,
	ICANDESCENCE,
	BUMP,
	DIFF_FACTOR,
	TRANSLUCENCE,
	TRANSLUCENCE_DEPTH,
	TRANSLUCENCE_FOCUS
};

// Texture Index are one to one with Uniform Index
enum class GFGMayaPhongLoc
{
	COLOR,
	TRANSPARENCY,
	AMBIENT_COLOR,
	ICANDESCENCE,
	BUMP,
	DIFF_FACTOR,
	TRANSLUCENCE,
	TRANSLUCENCE_DEPTH,
	TRANSLUCENCE_FOCUS,
	COSINE_POWER,
	SPECULAR_COLOR,
	REFLECTIV,
	REFLECTED_COLOR
};

// Texture Index are one to one with Uniform Index
enum class GFGMayaDX11UniformLoc
{
	DIFFUSE_TEXTURE_ON,
	DIFFUSE_COLOR,
	OPACITY,
	OPACITY_FRESNEL_MIN,
	OPACITY_FRESNEL_MAX,
	SPECULAR_TEXTURE_ON,
	SPECULAR_COLOR,
	SPECULAR_POWER,
	NORMAL_TEXTURE_ON,
	NORMAL_HEIGHT,
	REFLECTION_TEXTURE_ON,
	REFLECTION_INTENSITY,
	REFLECTION_BLUR,
	DISPLACEMENT_TEXTURE_ON,
	DISPLACEMENT_HEIGHT,
	DISPLACEMENT_OFFSET
};

enum class GFGMayaDX11TextureLoc
{
	DIFFUSE_TEXTURE,
	SPEUCLAR_TEXTURE,
	NORMAL_TEXTURE,
	REFLECTION_TEXTURE,
	DISPLACEMENT_TEXTURE
};

#endif //__GFG_MATERIALTYPES_H__