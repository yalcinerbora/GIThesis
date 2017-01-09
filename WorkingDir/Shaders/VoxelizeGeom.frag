#version 430
#extension GL_NV_gpu_shader5 : require
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_NV_shader_atomic_int64 : require
#extension GL_NV_shader_atomic_float : require
/*	
	**Voxelize Geom Shader**
	
	File Name	: VoxelizeGeom.frag
	Author		: Bora Yalciner
	Description	:

		Voxelizes Objects
*/

// Definitions
#define IN_UV layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_POS layout(location = 2)

#define LU_AABB layout(std430, binding = 3) readonly
#define LU_NORMAL_SPARSE layout(std430, binding = 6) coherent volatile
#define LU_COLOR_SPARSE layout(std430, binding = 7) coherent volatile
#define LU_VOXEL_DATA layout(std430, binding = 8) coherent volatile

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

LU_VOXEL_DATA buffer VoxelData
{
	uint64_t voxelData[];
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

uint64_t Average(in uint64_t current, in vec3 normal, in vec3 color, in float specular)
{
	uvec2 splitCurrent = unpackUint2x32(current);
	vec4 colorCur = unpackUnorm4x8(splitCurrent.x);
	vec4 normalCur = unpackUnorm4x8(splitCurrent.y);
	return 0;
}

//void AtomicAverage(in vec3 normal, in vec3 color, 
//				   in float specular, in ivec3 iCoord)
//{
//	uint coord = iCoord.z * texSize3D * texSize3D +
//				 iCoord.y * texSize3D +
//			 	 iCoord.x;
//	uint64_t current;
//	uint64_t previous = 0;
//	while((curent = atomicCompSwap(voxelData[coord], previous, new)) != previous)
//	{
//		previous = current;

//	}
//}

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

	vec4 color = texture2D(colorTex, fUV).rgba;

	if(iCoord.x < texSize3D &&
	   iCoord.y < texSize3D &&
	   iCoord.z < texSize3D &&
	   iCoord.x >= 0 &&
	   iCoord.y >= 0 &&
	   iCoord.z >= 0)
	{
		AtomicAverage(fNormal, color.rgb, color.a, iCoord);

		//// Non atomic overwrite version
		//normalSparse[iCoord.z * texSize3D * texSize3D +
		//			 iCoord.y * texSize3D +
		//			 iCoord.x] = vec4(fNormal, 1.0f);
		//colorSparse[iCoord.z * texSize3D * texSize3D +
		//			iCoord.y * texSize3D +
		//			iCoord.x] = vec4(color, 1.0f);
	}
}