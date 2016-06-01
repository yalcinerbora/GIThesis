#version 430
#extension GL_NV_shader_thread_group : require
#extension GL_NV_shader_thread_shuffle : require
/*	
	**Surface Raytrace Compute Shader**
	
	File Name	: SurfAO.vert
	Author		: Bora Yalciner
	Description	:

	Ambient Occlusion Approximation using surface based block fetch
		
*/
#define I_LIGHT_INENSITY layout(rgba8, binding = 2) restrict writeonly

#define LU_SVO_NODE layout(std430, binding = 0) readonly
#define LU_SVO_MATERIAL layout(std430, binding = 1) readonly
#define LU_SVO_LEVEL_OFFSET layout(std430, binding = 2) readonly

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)
#define U_SVO_CONSTANTS layout(std140, binding = 3)
#define U_CONE_PARAMS layout(std140, binding = 4)

#define T_NORMAL layout(binding = 1)
#define T_DEPTH layout(binding = 2)
#define T_DENSE_NODE layout(binding = 5)
#define T_DENSE_MAT layout(binding = 6)

#define FLT_MAX 1E+37
#define CONE_COUNT 4
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define WARP_SIZE 32

// Uniforms
uniform vec2 CONE_ORTHO[4] = 
{
	vec2( -1.0f, -1.0f),
    vec2( -1.0f, 1.0f),
    vec2( 1.0f, -1.0f),
    vec2( 1.0f, 1.0f)
};

uniform vec3 AABB_CORNER[8] = 
{
	vec3( 0.0f, 0.0f, 0.0f),
	vec3( 1.0f, 0.0f, 0.0f),
	vec3( 0.0f, 1.0f, 0.0f),
	vec3( 1.0f, 1.0f, 0.0f),
	vec3( 0.0f, 0.0f, 1.0f),
	vec3( 1.0f, 0.0f, 1.0f),
	vec3( 0.0f, 1.0f, 1.0f),
	vec3( 1.0f, 1.0f, 1.0f)
};

LU_SVO_NODE buffer SVONode
{ 
	uint svoNode[];
};

LU_SVO_MATERIAL buffer SVOMaterial
{ 
	uvec2 svoMaterial[];
};

LU_SVO_LEVEL_OFFSET buffer SVOLevelOffsets
{
	uint svoLevelOffset[];
};

U_SVO_CONSTANTS uniform SVOConstants
{
	// xyz gridWorldPosition
	// w is gridSpan
	vec4 worldPosSpan;

	// x is grid dimension
	// y is grid depth
	// z is dense dimension
	// w is dense depth
	uvec4 dimDepth;

	// x is cascade count
	// y is node sparse offet
	// z is material sparse offset
	// w is dense mat tex min level
	uvec4 offsetCascade;
};

U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
};

U_INVFTRANSFORM uniform InverseFrameTransform
{
	mat4 invViewProjection;

	vec4 camPos;		// To Calculate Eye
	vec4 camDir;		// To Calculate Eye
	ivec4 viewport;		// Viewport Params
	vec4 depthNearFar;	// depth range params (last two unused)
};

U_CONE_PARAMS uniform ConeTraceParameters
{
	// x max traverse distance
	// y tangent(coneAngle)
	// z tangent(coneAngle * 0.5f)
	// w sample ratio;
	vec4 coneParams1;

	// x intensity
	// y sqrt2
	// z sqrt3
	// w empty
	vec4 coneParams2;
};

// Textures
uniform I_LIGHT_INENSITY image2D liTex;

uniform T_NORMAL usampler2D gBuffNormal;
uniform T_DEPTH sampler2D gBuffDepth;
uniform T_DENSE_NODE usampler3D tSVODense;
uniform T_DENSE_MAT usampler3D tSVOMat;

// Surface Volume Min Max Reduction helper memory
// Actual packed surface values
shared uvec2 surfaceVolume[8];
shared vec3 surfMax, surfMin;
shared vec3 reduceBuffer[BLOCK_SIZE_X * BLOCK_SIZE_Y / WARP_SIZE];

// Functions
vec3 DepthToWorld(vec2 gBuffUV)
{
	// Converts Depthbuffer Value to World Coords
	// First Depthbuffer to Screen Space
	vec3 ndc = vec3(gBuffUV, texture(gBuffDepth, gBuffUV).x);
	ndc.xy = 2.0f * ndc.xy - 1.0f;
	ndc.z = ((2.0f * (ndc.z - depthNearFar.x) / (depthNearFar.y - depthNearFar.x)) - 1.0f);

	// Clip Space
	vec4 clip;
	clip.w = projection[3][2] / (ndc.z - (projection[2][2] / projection[2][3]));
	clip.xyz = ndc * clip.w;

	// From Clip Space to World Space
	return (invViewProjection * clip).xyz;
}

ivec3 LevelVoxId(in vec3 worldPoint, in uint depth)
{
	ivec3 result = ivec3(floor((worldPoint - worldPosSpan.xyz) / worldPosSpan.w));
	return result >> (dimDepth.y - depth);
}

vec3 LevelVoxIdF(in vec3 worldPoint, in uint depth)
{
	return (worldPoint - worldPosSpan.xyz) / (worldPosSpan.w * float((0x1 << (dimDepth.y - depth))));
}

vec3 VoxIdToWorld(in ivec3 voxId, in uint depth)
{
	return (vec3(voxId) + 0.5f) * worldPosSpan.w * 
			float(0x1 << (dimDepth.y - depth)) + worldPosSpan.xyz;
}

uint SpanToDepth(in uint number)
{
	return dimDepth.y - findMSB(number);
}

uint CalculateLevelChildId(in ivec3 voxPos, in uint levelDepth)
{
	uint bitSet = 0;
	bitSet |= ((voxPos.z >> (dimDepth.y - levelDepth)) & 0x000000001) << 2;
	bitSet |= ((voxPos.y >> (dimDepth.y - levelDepth)) & 0x000000001) << 1;
	bitSet |= ((voxPos.x >> (dimDepth.y - levelDepth)) & 0x000000001) << 0;
	return bitSet;
}

vec3 UnpackNormalGBuff(in uvec2 norm)
{
	vec3 result;
	result.x = ((float(norm.x) / 0xFFFF) - 0.5f) * 2.0f;
	result.y = ((float(norm.y & 0x7FFF) / 0x7FFF) - 0.5f) * 2.0f;
	result.z = sqrt(abs(1.0f - dot(result.xy, result.xy)));
	result.z *= sign(int(norm.y << 16));
	return result;
}

vec4 UnpackColorSVO(in uint colorPacked)
{
	return unpackUnorm4x8(colorPacked);
}

vec4 UnpackNormalSVO(in uint voxNormPosY)
{
	return vec4(unpackSnorm4x8(voxNormPosY).xyz,
		        unpackUnorm4x8(voxNormPosY).w);
}

vec3 GenInterpolPos(in vec3 worldPos, in uint interpolId, in uint depth)
{
	vec3 voxPos = LevelVoxIdF(worldPos, depth);
	vec3 voxPosFloor = vec3(LevelVoxId(worldPos, depth));
	ivec3 voxNeigbour = ivec3(round(voxPos - voxPosFloor)) * 2 - 1;
	ivec3 interpolLoc = LevelVoxId(worldPos, depth) + ivec3(AABB_CORNER[interpolId]) * voxNeigbour;
	return VoxIdToWorld(interpolLoc, depth);
}

vec3 GetInterpolWeights(in vec3 worldPos, in uint depth)
{
	vec3 voxPos = LevelVoxIdF(worldPos, depth);
	vec3 voxPosFloor = vec3(LevelVoxId(worldPos, depth));
	vec3 voxNeigbour = round(voxPos - voxPosFloor) * 2.0f - 1.0f;
	ivec3 bottomLeft = LevelVoxId(worldPos, depth) + min(ivec3(voxNeigbour), 0);
	vec3 blWorld = VoxIdToWorld(bottomLeft, depth);
	vec3 interpolWeights = (worldPos - blWorld) / (worldPosSpan.w * float((0x1 << (dimDepth.y - depth))));

	interpolWeights.x = (voxNeigbour.x < 0) ? (1.0f - interpolWeights.x) : interpolWeights.x;
	interpolWeights.y = (voxNeigbour.y < 0) ? (1.0f - interpolWeights.y) : interpolWeights.y;
	interpolWeights.z = (voxNeigbour.z < 0) ? (1.0f - interpolWeights.z) : interpolWeights.z;
	//interpolWeights.z = (voxNeigbour.z < 0) ? interpolWeights.z : (1.0f - interpolWeights.z); // RH Coord system
	//interpolWeights.x = (voxNeigbour.x < 0) ? interpolWeights.x : (1.0f - interpolWeights.x); // RH Coord system
	//interpolWeights.y = (voxNeigbour.y < 0) ? interpolWeights.y : (1.0f - interpolWeights.y); // RH Coord system
	return interpolWeights;
}

vec3 GetAABBInterpol(in vec3 aabbMin, in vec3 aabbMax, in vec3 worldPos, in uint depth)
{
	vec3 voxPos = LevelVoxIdF(worldPos, depth);
	vec3 voxPosFloor = vec3(LevelVoxId(worldPos, depth));
	vec3 voxNeigbour = round(voxPos - voxPosFloor) * 2.0f - 1.0f;
	vec3 interpolWeights = (worldPos - aabbMin) / (aabbMax - aabbMin);
	interpolWeights.x = (voxNeigbour.x < 0) ? (1.0f - interpolWeights.x) : interpolWeights.x;
	interpolWeights.y = (voxNeigbour.y < 0) ? (1.0f - interpolWeights.y) : interpolWeights.y;
	interpolWeights.z = (voxNeigbour.z < 0) ? (1.0f - interpolWeights.z) : interpolWeights.z;
	//interpolWeights.z = (voxNeigbour.z < 0) ? interpolWeights.z : (1.0f - interpolWeights.z); // RH Coord system
	return interpolWeights;
}

uvec2 FetchSVO(in vec3 worldPos, in uint interpolId, in uint depth)
{	
	// Generate Interpolation Position from worldPos
	worldPos = GenInterpolPos(worldPos, interpolId, depth);

	// Cull if out of bounds
	ivec3 voxPos = LevelVoxId(worldPos, dimDepth.y);
	if(any(lessThan(voxPos, ivec3(0))) ||
	   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
		return uvec2(0);

	// Tripolation is different if its sparse or dense
	// Fetch from 3D Tex here
	if(depth < offsetCascade.w)
	{
		// Not every voxel level is available
		return uvec2(0);
	}
	else if(depth <= dimDepth.w)
	{
		// Dense Fetch
		uint mipId = dimDepth.w - depth;
		uint levelDim = dimDepth.z >> mipId;
		vec3 levelUV = LevelVoxIdF(worldPos, depth) / float(levelDim);

		// Fetch according to your interpol id
		return textureLod(tSVOMat, levelUV, float(mipId)).xy;
	}
	else
	{
		// Sparse Fetch
		ivec3 denseVox = LevelVoxId(worldPos, dimDepth.w);
		vec3 texCoord = vec3(denseVox) / dimDepth.z;
		uint nodeIndex = texture(tSVODense, texCoord).x;

		if(nodeIndex == 0xFFFFFFFF) return uvec2(0);
		nodeIndex += CalculateLevelChildId(voxPos, dimDepth.w + 1);

		uint traversedLevel;
		for(traversedLevel = dimDepth.w + 1; traversedLevel < depth; traversedLevel++)
		{
			// Fetch Next Level
			uint levelOffset = offsetCascade.y + svoLevelOffset[traversedLevel - dimDepth.w];
			uint newNodeIndex = svoNode[levelOffset + nodeIndex];
			
			// Break if its the leaf of the node and use that value instead
			if(newNodeIndex == 0xFFFFFFFF) 
				//return uvec2(0x00000000); 
				//return uvec2(0xFF000000); 
				break;

			nodeIndex = newNodeIndex + CalculateLevelChildId(voxPos, traversedLevel + 1);
		}
		// At requested level or the level most deepest at that location
		uint matLoc = offsetCascade.z + svoLevelOffset[traversedLevel - dimDepth.w] + nodeIndex;
		return svoMaterial[matLoc];
	}

	//// Start tracing (stateless start from root (dense))
	//ivec3 voxPos = LevelVoxId(worldPos, dimDepth.y);

	//// Cull if out of bounds
	//if(any(lessThan(voxPos, ivec3(0))) ||
	//   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
	//	return uvec2(0);

	//// Tripolation is different if its sparse or dense
	//// Fetch from 3D Tex here
	//if(depth < offsetCascade.w)
	//{
	//	// Not every voxel level is available
	//	return uvec2(0);
	//}
	//else if(depth <= dimDepth.w)
	//{
	//	// Dense Fetch
	//	uint mipId = dimDepth.w - depth;
	//	uint levelDim = dimDepth.z >> mipId;
	//	vec3 levelUV = LevelVoxIdF(worldPos, depth) / float(levelDim);
	//	return textureLod(tSVOMat, levelUV, float(mipId)).xy;

	//}
	//else
	//{
	//	ivec3 denseVox = LevelVoxId(worldPos, dimDepth.w);
	//	vec3 texCoord = vec3(denseVox) / dimDepth.z;
	//	unsigned int nodeIndex = texture(tSVODense, texCoord).x;

	//	if(nodeIndex == 0xFFFFFFFF) return uvec2(0);
	//	nodeIndex += CalculateLevelChildId(voxPos, dimDepth.w + 1);

	//	for(uint i = dimDepth.w + 1; i < depth; i++)
	//	{
	//		// Fetch Next Level
	//		uint newNodeIndex = svoNode[offsetCascade.y + svoLevelOffset[i - dimDepth.w] + nodeIndex];

	//		// Node check
	//		// If valued node go deeper else return no occlusion
	//		if(newNodeIndex == 0xFFFFFFFF) return uvec2(0);
	//		else nodeIndex = newNodeIndex + CalculateLevelChildId(voxPos, i + 1);
	//	}
	//	// Finally At requested level
	//	// BackTrack From Child
	//	nodeIndex -= CalculateLevelChildId(voxPos, depth);
	//	uint matLoc = offsetCascade.z + svoLevelOffset[depth - dimDepth.w] +
	//				  nodeIndex;
	//	return svoMaterial[matLoc];
	//}
}

void GenSurfaceAABB(inout vec3 aabbMin, inout vec3 aabbMax, in uint localId)
{
	uint laneId = localId % gl_WarpSizeNV;
	uint warpId = localId / gl_WarpSizeNV;

	// Switch Depending on the component
	// Each warp will reduce 32 to 1
	for(uint offset = gl_WarpSizeNV / 2; offset > 0; offset /= 2)
	{
		vec3 neigMax = shuffleDownNV(aabbMax, offset, gl_WarpSizeNV);
		vec3 neigMin = shuffleDownNV(aabbMin, offset, gl_WarpSizeNV);

		aabbMin = min(neigMin, aabbMin);
		aabbMax = max(neigMax, aabbMax);
	}

	// Leader of each warp (in this case lane 0) write to sMem
	// Once for min
	if(laneId == 0) reduceBuffer[warpId] = aabbMin;
	barrier();
	aabbMin = (localId < (BLOCK_SIZE_X * BLOCK_SIZE_Y) / gl_WarpSizeNV) ? reduceBuffer[laneId] : vec3(FLT_MAX);
	barrier();
	// Once for max
	if(laneId == 0) reduceBuffer[warpId] = aabbMax;
	barrier();
	aabbMax = (localId < (BLOCK_SIZE_X * BLOCK_SIZE_Y) / gl_WarpSizeNV) ? reduceBuffer[laneId] : vec3(-FLT_MAX);

	// Last Reduction using first warp
	if(warpId == 0)
	{
		for(uint offset = gl_WarpSizeNV / 2; offset > 0; offset /= 2)
		{
			vec3 neigMax = shuffleDownNV(aabbMax, offset, gl_WarpSizeNV);
			vec3 neigMin = shuffleDownNV(aabbMin, offset, gl_WarpSizeNV);

			aabbMin = min(neigMin, aabbMin);
			aabbMax = max(neigMax, aabbMax);
		}
	}

	if(localId == 0)
	{
		surfMax = aabbMax;
		surfMin = aabbMin;
	}
	barrier();

	// All Done!
	aabbMax = surfMax;
	aabbMin = surfMin;
}
	
void GenAABBValues(in uvec2 nodeData, in vec3 interpolWeights, in uint localId, in uint depth)
{
	uint laneId = localId % gl_WarpSizeNV;
	uint warpId = localId / gl_WarpSizeNV;

	// Use Warp Instructions to reduce values for each node then Interpolate
	// we have 2warps responsible for fetching data and using it
	if(warpId <= 1)
	{
		for(int offset = 4; offset > 0; offset /= 2)
		{
			uvec2 neigNode = shuffleDownNV(nodeData, offset, 8);
			
			vec4 color = UnpackColorSVO(nodeData.x);
			vec4 normal = UnpackNormalSVO(nodeData.y);

			vec4 colorNeig = UnpackColorSVO(neigNode.x);
			vec4 normalNeig = UnpackNormalSVO(neigNode.y);

			// Occlusion values are not valid on the last depth since that value
			// Used as a averaging count
			if(offset == 4 && depth == dimDepth.y)
			{
				normal.w = ceil(normal.w);
				normalNeig.w = ceil(normalNeig.w);
			}
			// Mix according each axis
			int interpolAxis = int(log2(offset));
			color = mix(color, colorNeig, interpolWeights[interpolAxis]);
			normal = mix(normal, normalNeig, interpolWeights[interpolAxis]);
			
			nodeData.x = packUnorm4x8(color);
			nodeData.y = (packSnorm4x8(normal) & 0x00FFFFFF) | 
						 (packUnorm4x8(normal) & 0xFF000000);
		}
		if(localId % 8 == 0) surfaceVolume[localId / 8] = nodeData;
	}
}

void Interpolate(out vec4 color, out vec4 normal, in vec3 interpValue)
{
	// Interp Color
	vec4 colorA, colorB, colorC, colorD, colorE, colorF, colorG, colorH;
	colorA = UnpackColorSVO(surfaceVolume[0].x);
	colorB = UnpackColorSVO(surfaceVolume[1].x); 
	colorC = UnpackColorSVO(surfaceVolume[2].x);
	colorD = UnpackColorSVO(surfaceVolume[3].x); 
	colorE = UnpackColorSVO(surfaceVolume[4].x); 
	colorF = UnpackColorSVO(surfaceVolume[5].x); 
	colorG = UnpackColorSVO(surfaceVolume[6].x); 
	colorH = UnpackColorSVO(surfaceVolume[7].x);

	colorA = mix(colorA, colorB, interpValue.x);
	colorB = mix(colorC, colorD, interpValue.x);
	colorC = mix(colorE, colorF, interpValue.x);
	colorD = mix(colorG, colorH, interpValue.x);

	colorA = mix(colorA, colorB, interpValue.y);
	colorB = mix(colorC, colorD, interpValue.y);

	color = mix(colorA, colorB, interpValue.z);
	
	vec4 normalA, normalB, normalC, normalD, normalE, normalF, normalG, normalH;
	normalA = UnpackNormalSVO(surfaceVolume[0].y);
	normalB = UnpackNormalSVO(surfaceVolume[1].y); 
	normalC = UnpackNormalSVO(surfaceVolume[2].y);
	normalD = UnpackNormalSVO(surfaceVolume[3].y); 
	normalE = UnpackNormalSVO(surfaceVolume[4].y); 
	normalF = UnpackNormalSVO(surfaceVolume[5].y); 
	normalG = UnpackNormalSVO(surfaceVolume[6].y); 
	normalH = UnpackNormalSVO(surfaceVolume[7].y);

	normalA = mix(normalA, normalB, interpValue.x);
	normalB = mix(normalC, normalD, interpValue.x);
	normalC = mix(normalE, normalF, interpValue.x);
	normalD = mix(normalG, normalH, interpValue.x);

	normalA = mix(normalA, normalB, interpValue.y);
	normalB = mix(normalC, normalD, interpValue.y);

	normal = mix(normalA, normalB, interpValue.z);
}

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	//if(any(greaterThanEqual(globalId, imageSize(liTex).xy))) return;

	// Fetch GBuffer and Interpolate Positions (if size is smaller than current gbuffer)
	vec2 gBuffUV = vec2(globalId + vec2(0.5f) - viewport.xy) / viewport.zw;
	vec3 worldPos = DepthToWorld(gBuffUV);
	vec3 worldNorm = UnpackNormalGBuff(texture(gBuffNormal, gBuffUV).xy);

	// Determine cascade no from distance of the camera
	// And the min span for that cascade
	vec3 gridCenter = worldPosSpan.xyz + worldPosSpan.w * (dimDepth.x >> 1);
	vec3 diff = abs(worldPos - gridCenter) / (worldPosSpan.w * (dimDepth.x >> offsetCascade.x));
	uvec3 cascades = findMSB(uvec3(diff)) + 1;
	uint cascadeNo = uint(max(cascades.x, max(cascades.y, cascades.z)));
	float cascadeSpan = worldPosSpan.w * (0x1 << cascadeNo);
	
	// Find Edge vectors from normal
	// [(-z-y) / x, 1, 1] is perpendicular (unless normal is X axis)
	// handle special case where normal is (1.0f, 0.0f, 0.0f)
	vec3 ortho1 = normalize(vec3(-(worldNorm.z + worldNorm.y) / worldNorm.x, 1.0f, 1.0f));
	ortho1 = mix(ortho1, vec3(0.0f, 1.0f, 0.0f), floor(worldNorm.x));
	vec3 ortho2 = normalize(cross(worldNorm, ortho1));

	// Cone Directions
	vec3 coneA = normalize(worldNorm + 
						   ortho1 * coneParams1.y * CONE_ORTHO[0].x + 
						   ortho2 * coneParams1.y * CONE_ORTHO[0].y);
	vec3 coneB = normalize(worldNorm + 
						   ortho1 * coneParams1.y * CONE_ORTHO[1].x + 
						   ortho2 * coneParams1.y * CONE_ORTHO[1].y);
	vec3 coneC = normalize(worldNorm + 
						   ortho1 * coneParams1.y * CONE_ORTHO[2].x + 
						   ortho2 * coneParams1.y * CONE_ORTHO[2].y);
	vec3 coneD = normalize(worldNorm + 
						   ortho1 * coneParams1.y * CONE_ORTHO[3].x + 
						   ortho2 * coneParams1.y * CONE_ORTHO[3].y);
	
	// Initally start the pos away from the surface since 
	// voxel system and polygon system are not %100 aligned
	worldPos += worldNorm * cascadeSpan * coneParams2.z * 4.0f;

	// Start sampling towards cone direction
	// Loop Traverses until MaxDistance Exceeded
	// March distance is variable per iteration
	float totalConeOcclusion = 0.0f;
	float marchDistance = cascadeSpan;
	for(float traversedDistance = cascadeSpan;
		traversedDistance <= coneParams1.x;
		traversedDistance += marchDistance)
	{
		// Pixel Pyramid Min Max
		vec3 pMin = vec3(FLT_MAX, FLT_MAX, FLT_MAX);
		vec3 pMax = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		pMin = min(pMin, traversedDistance * coneA + worldPos);
		pMin = min(pMin, traversedDistance * coneB + worldPos);
		pMin = min(pMin, traversedDistance * coneC + worldPos);
		pMin = min(pMin, traversedDistance * coneD + worldPos);
		
		pMax = max(pMax, traversedDistance * coneA + worldPos);
		pMax = max(pMax, traversedDistance * coneB + worldPos);
		pMax = max(pMax, traversedDistance * coneC + worldPos);
		pMax = max(pMax, traversedDistance * coneD + worldPos);

		//pMin = traversedDistance * worldNorm + worldPos;
		//pMax = traversedDistance * worldNorm + worldPos;

		// Reduce this between threads
		uint localId = gl_LocalInvocationID.y * BLOCK_SIZE_X + gl_LocalInvocationID.x;
		GenSurfaceAABB(pMin, pMax, localId);

		// Current Cone Sample Diameter and its corresponding depth
		// diameter will cover entire AABB box
		// using its half diagonal as diameter
		float diameter = length(pMax - pMin) * 0.25f;
		//float diameter = max(cascadeSpan, coneParams1.z * 2.0f * traversedDistance);
		uint nodeDepth = SpanToDepth(uint(round(diameter / worldPosSpan.w)));
		//float diameter = 10;
		//uint nodeDepth = 6;

		// pMin pMax Defines an AABB now
		// Determine your aabb corner and interpolation node
		// AABB has 8 corners and each corner has 8 interpolation points
		// which will be fetched from SVO Tree
		if(localId < 8 * 8)
		{
			uint cornerId = localId / 8;
			uint interpolId = localId % 8;

			// Generate your Corner
			vec3 aabbCorner = AABB_CORNER[cornerId] * pMax +
							  (vec3(1.0f) - AABB_CORNER[cornerId]) * pMin;
			//vec3 aabbCorner = pMin;

			// Fetch data from SVO Tree
			uvec2 nodeData = FetchSVO(aabbCorner, interpolId, nodeDepth);
			//nodeData.y = min(0x7F000000, nodeData.y);

			////nodeData.y = 0;
			//if((nodeData.y & 0xFF000000) == 0xFF000000)
			//	nodeData.y = 0;
			
			// Interpolate corner
			// Write Interpolated results to the shared memory
			vec3 weights = GetInterpolWeights(worldPos, nodeDepth);
			//vec3 weights = vec3(0.5f);
			GenAABBValues(nodeData, weights, localId, nodeDepth);
		}
		// This Barrier Important required by GenAABB cant write this inside a scope
		barrier();

		// Generated AABB corners
		// Now we can fetch 3D interpolate sample
		float surfOcclusion = 0.0f;
		for(uint i = 0; i < CONE_COUNT; i++)
		//for(uint i = 1; i < 2; i++)
		{
			// Get a Cone Center Point
			vec3 cone = normalize(worldNorm + 
								  ortho1 * coneParams1.z * CONE_ORTHO[i].x + 
								  ortho2 * coneParams1.z * CONE_ORTHO[i].y);
			//vec3 cone = worldNorm;
			vec3 samplePos = worldPos + traversedDistance * cone;

			// Interpolation of the point
			vec4 color, normal;
			vec3 interpValue = GetAABBInterpol(pMin, pMax, samplePos, nodeDepth);
			//vec3 interpValue = vec3(0.5f);
			interpValue = min(max(interpValue, 0.0f), 1.0f);
			Interpolate(color, normal, interpValue);

			// We use only AO here since this shader calculates AO only
			surfOcclusion += dot(worldNorm, cone) * normal.w * 0.25f;// * 2.0f;
		}

		// do AO calculations from this value (or values)
		// Correction Term to prevent intersecting samples error
		float diameterVoxelSize = worldPosSpan.w * (0x1 << (dimDepth.y - nodeDepth));
		surfOcclusion = 1.0f - pow(1.0f - surfOcclusion, marchDistance / diameterVoxelSize);

		// Occlusion falloff (linear)
		surfOcclusion *= (1.0f / (1.0f + traversedDistance));
		//surfOcclusion *= (1.0f / (1.0f + pow(traversedDistance, 0.5f)));

		// Average total occlusion value
		totalConeOcclusion += (1 - totalConeOcclusion) * surfOcclusion;
		
		// Advance sample point (from sampling diameter)
		marchDistance = diameter * coneParams1.w;
	}

	// Cos tetha multiplication
	totalConeOcclusion *= coneParams2.x;
	
	// Store result
	vec4 result = vec4(totalConeOcclusion);
	//if(any(lessThan(globalId, imageSize(liTex).xy)))
	//	imageStore(liTex, ivec2(globalId), 1.0f - result);
	//	//imageStore(liTex, ivec2(globalId), vec4(pMin, 1.0f));

	if(gl_WorkGroupID.x == 0 && gl_WorkGroupID.y == 0)
	{
		//if(gl_LocalInvocationID.x == 0 ||
		//	gl_LocalInvocationID.x == 15 ||
		//	gl_LocalInvocationID.y == 0 ||
		//	gl_LocalInvocationID.y == 15)
		imageStore(liTex, ivec2(globalId), vec4(1.0f, 0.0f, 0.0f, 0.0f));
	}
	else
	{
		if(any(lessThan(globalId, imageSize(liTex).xy)))
			imageStore(liTex, ivec2(globalId), 1.0f - result);
	}
}