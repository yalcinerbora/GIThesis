#version 430
/*	
	**G-Buffer Material Shader**
	
	File Name	: AGenericMaterial.frag 
	Author		: Bora Yalciner
	Description	:

		Simple Normal Map Color Map Material
		Has UV factor and offsets
		Has to be appended and compiled at the end of the "GWriteGeneric.frag" shader
		in order to be used
*/

// Definitions
#define LU_MATERIAL0 layout(binding = 0)
#define LU_MATERIAL1 layout(binding = 1)
#define LU_MATERIAL2 layout(binding = 2)	// 3-4 Reserved (3 Used as Render Data 4 used for shrink)
#define LU_MATERIAL3 layout(binding = 5)

#define U_MATERIAL0 layout(binding = 0)
#define U_MATERIAL1 layout(binding = 1)
#define U_MATERIAL1 layout(binding = 2)

#define T_COLOR layout(binding = 0)
#define T_NORMAL layout(binding = 1)

// Textures
uniform T_COLOR sampler2DArray colorTexArray;
uniform T_NORMAL sampler2DArray normalTexArray;

// Uniforms
LU_MATERIAL0 buffer MaterialData
{
	struct
	{
		vec4 normalUVOffsetFactor;	// first 2 offset last 2 factor
		vec4 colorUVOffsetFactor;
	} perMaterialData[];
};

void GBufferPopulate(out vec3 fNormal, out vec3 fColor, out vec2 metalSpecular)
{
	// This One Uses Normal Mapping, With Basic Color Map
	// Normal
	// Fetch Normal Map output it as view space

	// LS 16 bit shows material Instance Index (MS 16 bit shows materialID)
	uint materialID = fMatID & 0x0000FFFF;

	// Normal
	// Apply Offsets Etc
	vec3 normalUV;
	normalUV.xy = fUV * perMaterialData[materialID].normalUVOffsetFactor.zw + perMaterialData[materialID].normalUVOffsetFactor.xy;
	normalUV.z = float(materialID);
	vec3 normal = (texture(normalTexArray, normalUV) * 2.0f - 1.0f).xyz;
	fNormal = fTBN * normal;

	// Color
	// Fetch Color Map
	vec3 colorUV;
	colorUV.xy = fUV * perMaterialData[materialID].colorUVOffsetFactor.zw + perMaterialData[materialID].colorUVOffsetFactor.xy;
	colorUV.z = float(materialID);
	vec3 fcolor = texture(colorTexArray, colorUV).rgb;

	// Metal Specular
	metalSpecular = vec2(0.0f, 0.0f);
}