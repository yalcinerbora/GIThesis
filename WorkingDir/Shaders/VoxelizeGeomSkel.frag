#version 430
#extension GL_NV_shader_atomic_float : require
/*	
	**Voxelize Geom Shader**
	
	File Name	: VoxelizeGeomSkeletal.frag
	Author		: Bora Yalciner
	Description	:

		Voxelizes Skeletal Objects
*/

// Definitions
#define IN_UV layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_POS layout(location = 2)
#define IN_WEIGHT layout(location = 4)
#define IN_WEIGHT_INDEX layout(location = 5)

#define LU_AABB layout(std430, binding = 3) readonly
#define LU_NORMAL_SPARSE layout(std430, binding = 6) coherent volatile
#define LU_COLOR_SPARSE layout(std430, binding = 7) coherent volatile
#define LU_WEIGHT_SPARSE layout(std430, binding = 8) coherent volatile

#define I_LOCK layout(r32ui, binding = 0) coherent volatile

#define T_COLOR layout(binding = 0)

#define U_SPAN layout(location = 1)
#define U_SEGMENT_SIZE layout(location = 2)
#define U_SPLIT_CURRENT layout(location = 7)
#define U_OBJ_ID layout(location = 4)
#define U_TEX_SIZE layout(location = 8)

// Input
in IN_UV vec2 fUV;
in IN_NORMAL vec3 fNormal;
in IN_POS vec3 fPos;
flat in IN_WEIGHT vec4 fWeight;
flat in IN_WEIGHT_INDEX uvec4 fWIndex;

// Images
uniform I_LOCK uimage3D lock;

// Textures
uniform T_COLOR sampler2D colorTex;

// Uniform Constants
U_SPAN uniform float span;
U_SEGMENT_SIZE uniform float segmentSize;
U_SPLIT_CURRENT uniform uvec3 currentSplit;
U_OBJ_ID uniform uint objId;
U_TEX_SIZE uniform uint texSize3D;

// Shader Torage
LU_AABB buffer AABB
{
	struct
	{
		vec4 aabbMin;
		vec4 aabbMax;
	} objectAABBInfo[];
};

LU_COLOR_SPARSE buffer ColorBuffer 
{
	vec4 colorSparse[];
};

LU_NORMAL_SPARSE buffer NormalBuffer 
{
	vec4 normalSparse[];
};

LU_WEIGHT_SPARSE buffer WeightBuffer 
{
	uvec2 weightSparse[];
};

void Average(in vec3 normal, in vec3 color, 
			 in float specular, in ivec3 iCoord)
{
	// Load
	vec4 avgNormal = normalSparse[iCoord.z * texSize3D * texSize3D +
								  iCoord.y * texSize3D +
								  iCoord.x];
	vec4 avgColor = colorSparse[iCoord.z * texSize3D * texSize3D +
								iCoord.y * texSize3D +
								iCoord.x];
	
	// Average Normal.w holds count
	avgNormal.xyz *=  avgNormal.w;
	avgColor *= avgNormal.w;

	avgNormal.xyz += fNormal;
	avgColor.xyz += color;
	avgColor.w += specular;

	float denom = 1.0f / (avgNormal.w + 1.0f);
	avgNormal.xyz *= denom;
	avgColor *= denom;
	avgNormal.w += 1.0f;
	
	// Write
	normalSparse[iCoord.z * texSize3D * texSize3D +
				 iCoord.y * texSize3D +
			 	 iCoord.x] = avgNormal;
	colorSparse[iCoord.z * texSize3D * texSize3D +
				iCoord.y * texSize3D +
				iCoord.x] = avgColor;
}

uint PackWeight(vec4 weights)
{
	return packUnorm4x8(weights);
}

uint PackWIndex(uvec4 wIndices)
{
	uint result = 0x00000000;
	result |= wIndices.w << 24 & 0xFF000000;
	result |= wIndices.z << 16 & 0x00FF0000;
	result |= wIndices.y <<  8 & 0x0000FF00;
	result |= wIndices.x <<  0 & 0x000000FF;
	return result;
}

void AtomicAverage(in vec3 normal, in vec3 color, 
				   in float specular, in ivec3 iCoord)
{	
	uint coord = iCoord.z * texSize3D * texSize3D +
				 iCoord.y * texSize3D +
			 	 iCoord.x;

	// Thanks nvidia Kappa
	atomicAdd(normalSparse[coord].x, normal.x);
	atomicAdd(normalSparse[coord].y, normal.y);
	atomicAdd(normalSparse[coord].z, normal.z);
	atomicAdd(normalSparse[coord].w, 1.0f);

	atomicAdd(colorSparse[coord].x, color.x);
	atomicAdd(colorSparse[coord].y, color.y);
	atomicAdd(colorSparse[coord].z, color.z);
	atomicAdd(colorSparse[coord].w, specular);
}

void main(void)
{
	// interpolated object space pos
	vec3 aabbMin = objectAABBInfo[objId].aabbMin.xyz;
	aabbMin += vec3(currentSplit) * vec3(segmentSize);
	vec3 voxelCoord = floor((fPos - aabbMin) / span);
	ivec3 iCoord = ivec3(voxelCoord);

	vec3 color = texture2D(colorTex, fUV).rgb;

	if(iCoord.x < texSize3D &&
	   iCoord.y < texSize3D &&
	   iCoord.z < texSize3D &&
	   iCoord.x >= 0 &&
	   iCoord.y >= 0 &&
	   iCoord.z >= 0)
	{
		AtomicAverage(fNormal, color, 1.0f, iCoord);

		// Weight write
		uvec2 weights;
		weights.x = PackWeight(fWeight);
		weights.y = PackWIndex(fWIndex);
		weightSparse[iCoord.z * texSize3D * texSize3D +
					 iCoord.y * texSize3D +
					 iCoord.x] = weights;

		//// Non atomic overwrite version
		//normalSparse[iCoord.z * texSize3D * texSize3D +
		//			 iCoord.y * texSize3D +
		//			 iCoord.x] = vec4(fNormal, 1.0f);
		//colorSparse[iCoord.z * texSize3D * texSize3D +
		//			iCoord.y * texSize3D +
		//			iCoord.x] = vec4(color, 1.0f);
	}
}