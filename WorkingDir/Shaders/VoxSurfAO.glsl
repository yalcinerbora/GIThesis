#version 430
/*	
	**Voxel Ambient Occulusion Compute Shader**
	
	File Name	: VoxSurfAO.vert
	Author		: Bora Yalciner
	Description	:

		Ambient Occulusion approximation using SVO
		Uses Surface tracing in order to interpolate
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

#define CONE_COUNT 4
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8
#define SURF_DIM_XY 3 // 3 * 3 coverage points

// Uniforms
uniform vec2 CONE_ORTHO[4] = 
{
	vec2( -1.0f, -1.0f),
    vec2( -1.0f, 1.0f),
    vec2( 1.0f, -1.0f),
    vec2( 1.0f, 1.0f)
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
	// w is renderLevel
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

// Surfaces traced by each pixel
shared uvec2 surface [BLOCK_SIZE_Y][(BLOCK_SIZE_X / CONE_COUNT) * SURF_DIM_XY * SURF_DIM_XY];

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

vec3 UnpackColor(in uint colorPacked)
{
	//vec3 color;
	//color.x = float((colorPacked & 0x000000FF) >> 0) / 255.0f;
	//color.y = float((colorPacked & 0x0000FF00) >> 8) / 255.0f;
	//color.z = float((colorPacked & 0x00FF0000) >> 16) / 255.0f;
	//return color;
	return unpackUnorm4x8(colorPacked).xyz;
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

vec3 UnpackNormalSVO(in uint voxNormPosY)
{
	vec3 result;
	result.x = ((float(voxNormPosY & 0xFFFF) / 0xFFFF) - 0.5f) * 2.0f;
	result.y = ((float((voxNormPosY >> 16) & 0x7FFF) / 0x7FFF) - 0.5f) * 2.0f;
	result.z = sqrt(abs(1.0f - dot(result.xy, result.xy)));
	result.z *= sign(int(voxNormPosY));
	
	return result;
}

float UnpackOcclusion(in uint colorPacked)
{
	return float((colorPacked & 0xFF000000) >> 24) / 255.0f;
}

void SampleSurface()
{
	// Samples the Surface
	// Each thread responsible for 2 nodes
	// 
}

// SVO Fetch
float FetchSVOOcclusion(in vec3 worldPos, in uint depth)
{	
	// Start tracing (stateless start from root (dense))
	ivec3 voxPos = LevelVoxId(worldPos, dimDepth.y);

	// Cull if out of bounds
	if(any(lessThan(voxPos, ivec3(0))) ||
	   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
		return 0;

	// Tripolation is different if its sparse or dense
	// Fetch from 3D Tex here
	if(depth <= dimDepth.w)
	{
		// TODO:
		return 0.0f;
	}
	else
	{
		ivec3 denseVox = LevelVoxId(worldPos, dimDepth.w);
		uint nodeIndex = svoNode[denseVox.z * dimDepth.z * dimDepth.z +
								 denseVox.y * dimDepth.z + 
								 denseVox.x];
		if(nodeIndex == 0xFFFFFFFF) return 0.0f;
		nodeIndex += CalculateLevelChildId(voxPos, dimDepth.w + 1);

		for(uint i = dimDepth.w + 1; i < depth; i++)
		{
			// Fetch Next Level
			uint newNodeIndex = svoNode[offsetCascade.y + svoLevelOffset[i - dimDepth.w] + nodeIndex];

			// Node check
			// If valued node go deeper else return no occlusion
			if(newNodeIndex == 0xFFFFFFFF) return 0.0f;
			else nodeIndex = newNodeIndex + CalculateLevelChildId(voxPos, i + 1);
		}
		// Finally At requested level
		uint matLoc = offsetCascade.z + svoLevelOffset[depth - dimDepth.w] + nodeIndex;
		if(depth != dimDepth.y)
			return UnpackOcclusion(svoMaterial[matLoc].x);
		else
		{
			float occ = UnpackOcclusion(svoMaterial[matLoc].x);
			return ceil(occ);
		}
	}
}

void SumPixelData(inout vec4 coneColorOcc)
{
	// Use Surface Shared Mem since allocation may reduce occupancy
	// (probably will reduce on kepler (or previous) cards we are on around 6kb limit)

	// Transactions are between warp level adjacent values 
	// so barrier shouldnt be necessary
	uvec2 localId = gl_LocalInvocationID.xy;
	uvec2 pixId = uvec2(localId.y, localId.x / CONE_COUNT);
	uint pixelConeId = localId.x % CONE_COUNT;

	// left ones share their data
	if(pixelConeId >= (CONE_COUNT / 2))
	{
		surface[pixId.y][pixId.x][pixelConeId - (CONE_COUNT / 2)] = packUnorm4x8(coneColorOcc);
	}
	
	// right ones reduce
	if(pixelConeId < (CONE_COUNT / 2)) 
	{
		// Lerp it at the middle (weighted avg)
		vec4 neigbour = unpackUnorm4x8(surface[pixId.y][pixId.x][pixelConeId]);
		coneColorOcc = mix(coneColorOcc, neigbour, 0.5f);
	}

	if(pixelConeId == 1)
		surface[pixId.y][pixId.x][0] = packUnorm4x8(coneColorOcc);
	
	if(pixelConeId == 0) 
	{
		vec4 neigbour = unpackUnorm4x8(surface[pixId.y][pixId.x][0]);
		coneColorOcc = mix(coneColorOcc, neigbour, 0.5f);
	}
}

void StoreSurface(in vec3 edgeMin, in vec3 edgeMax)
{
	//uint nodeDepth = SpanToDepth(uint(round(diameter / worldPosSpan.w)));;
	//float depthMultiplier =  0x1 << (dimDepth.y - nodeDepth);
	//float currentVoxelSize = worldPosSpan.w * depthMultiplier;

	// Fetch Surface
	// Each Pixel needs to fetch 3x3 coverage points
	// 4 threads used for each pixel
	// 2 for each pixel + 1
	// Always fetch surfaces bigger than cone coverage
	// Surfaces should be flat for better sampling and consistency between pixels
	// Surfaces should be higher than the actual point in order to interpolate accrately

	// uvec2 each holds unom8 types
	// first uint holds color + occlusion
	// second uint holds normal


	// Atm can only store 2 vecs (4 on maxwell)
	//

	// Sample Depth Buffers of each light that will be involved in GI
	// Calculate Light Intensity of each coverage point and store

}

float SampleSurface(in vec3 position)
{
	// Determine ratios of the sampling points
	// Average samples according to this position
	return 0.0f;			
}

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per cone per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	uvec2 pixelId = globalId / uvec2(CONE_COUNT, 1);
	if(any(greaterThanEqual(pixelId, imageSize(liTex).xy))) return;

	// Fetch GBuffer and Interpolate Positions (if size is smaller than current gbuffer)
	vec2 gBuffUV = vec2(pixelId + vec2(0.5f) - viewport.xy) / viewport.zw;
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

	// Find Corner points of the surface
	vec3 edgeMin = normalize(worldNorm - ortho1 * coneParams1.y - ortho2 * coneParams1.y);
	vec3 edgeMax = normalize(worldNorm + ortho1 * coneParams1.y + ortho2 * coneParams1.y);
	vec3 coneDir = normalize(worldNorm + 
							 ortho1 * coneParams1.z * CONE_ORTHO[globalId.x % CONE_COUNT].x + 
							 ortho2 * coneParams1.z * CONE_ORTHO[globalId.x % CONE_COUNT].y);

	// Previous surface point and occlusion data
	float totalConeOcclusion = 0.0f;
	float prevOcclusion = 0.0f;
	float prevSurfPoint = 0.0f;

	// Initally Start the cone away from the surface since 
	// voxel system and polygon system are not %100 aligned
	worldPos += coneDir * cascadeSpan * coneParams2.z;

	// Start sampling towards that direction
	// Loop Traverses until MaxDistance Exceeded
	// March distance is variable per iteration
	float marchDistance = cascadeSpan;
	for(float traversedDistance = cascadeSpan;
		traversedDistance <= coneParams1.x;
		traversedDistance += marchDistance)
	{
		// Current Cone Sample Diameter
		// and its corresponding depth
		float diameter = max(cascadeSpan, coneParams1.z * 2.0f * traversedDistance);
		uint nodeDepth = SpanToDepth(uint(round(diameter / worldPosSpan.w)));

		// Determine Coverage Span of the surface 
		// (wrt cone angle and distance from pixel)
		// And Store 3x3 voxels
		float surfacePoint = (traversedDistance + diameter * 0.5f);
		vec3 surfMin = edgeMin * surfacePoint;
		vec3 surfMax = edgeMax * surfacePoint;
		StoreSurface(surfMin, surfMax);
		
		// start sampling from that surface (interpolate)
		//float surfOcclusion = SampleSurface(coneDir * traversedDistance);
		float surfOcclusion = FetchSVOOcclusion(worldPos + coneDir * traversedDistance,
												nodeDepth);

		// Omit if %100 occuluded in closer ranges
		// Since its not always depth pos aligned with voxel pos
	//	bool isOmitDistance = (surfOcclusion > 0.9f) && (traversedDistance < (coneParams2.z * worldPosSpan.w * (0x1 << offsetCascade.x - 1)));
	//	surfOcclusion = isOmitDistance ? 0.0f : surfOcclusion;		

		// than interpolate with your previous surface's value to simulate quadlinear interpolation
		float ratio = (traversedDistance - prevSurfPoint) / (surfacePoint - prevSurfPoint);
		float nodeOcclusion = mix(prevOcclusion, surfOcclusion, ratio);
		//float nodeOcclusion = surfOcclusion;
		
		// do AO calculations from this value (or values)
		// Correction Term to prevent intersecting samples error
		float diameterVoxelSize = worldPosSpan.w * (0x1 << (dimDepth.y - nodeDepth));
		nodeOcclusion = 1.0f - pow(1.0f - nodeOcclusion, marchDistance / diameterVoxelSize);
		
		// Occlusion falloff (linear)
		nodeOcclusion *= (1.0f / (1.0f + traversedDistance));
		//nodeOcclusion *= (1.0f / (1.0f + pow(traversedDistance, 0.5f)));

		// Average total occlusion value
		totalConeOcclusion += (1 - totalConeOcclusion) * nodeOcclusion;

		// Store Current Surface values as previous values
		prevOcclusion = surfOcclusion;
		prevSurfPoint = surfacePoint;

		// Advance sample point (from sampling diameter)
		marchDistance = diameter * coneParams1.w;
	}
	// Cos tetha multiplication
	totalConeOcclusion *= dot(worldNorm, coneDir) * coneParams2.x;
	
	// Sum occlusion data
	// CosTetha multiplicatio
	vec4 result = vec4(totalConeOcclusion);
	SumPixelData(result);
	
	// All Done!
	if(globalId.x % CONE_COUNT == 0) 
		imageStore(liTex, ivec2(pixelId), 1.0f - result);
		////imageStore(liTex, ivec2(pixelId), vec4(vec3(1.0f - totalConeOcclusion), 0.0f));
		//imageStore(liTex, ivec2(pixelId), vec4(worldPos, 0.0f));
		////imageStore(liTex, ivec2(pixelId), vec4(worldNorm, 0.0f));
	
}
