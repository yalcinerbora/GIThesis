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

#define U_MAX_DISTANCE layout(location = 0)
#define U_CONE_ANGLE layout(location = 1)
#define U_SAMPLE_DISTANCE layout(location = 2)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)
#define U_SVO_CONSTANTS layout(std140, binding = 3)

#define T_NORMAL layout(binding = 1)
#define T_DEPTH layout(binding = 2)

#define THREAD_PER_PIX 4
#define BLOCK_SIZE_XY 16
#define SAMPLE_PER_PIXEL_XY 3 // 3 * 3 coverage points

// 





uniform vec2 EDGE_ID_MAP[ 4 ] = 
{
	vec2( 0.0f, 0.0f),
    vec2( 0.0f, 1.0f),
    vec2( 1.0f, 0.0f),
    vec2( 1.0f, 1.0f)
};



// Surfaces traced by each pixel
shared uvec2 surface [(BLOCK_SIZE_XY / 2) * SAMPLE_PER_PIXEL_XY]
					 [(BLOCK_SIZE_XY / 2) * SAMPLE_PER_PIXEL_XY];

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
		return 0;
	}
	else
	{
		ivec3 denseVox = LevelVoxId(worldPos, dimDepth.w);
		uint nodeIndex = svoNode[denseVox.z * dimDepth.z * dimDepth.z +
								 denseVox.y * dimDepth.z + 
								 denseVox.x];
		if(nodeIndex == 0xFFFFFFFF) return 0;
		nodeIndex += CalculateLevelChildId(voxPos, dimDepth.w + 1);
		for(uint i = dimDepth.w + 1; i < depth; i++)
		{
			// Fetch Next Level
			uint newNodeIndex = svoNode[offsetCascade.y +
										svoLevelOffset[i - dimDepth.w] +
										nodeIndex];

			// Node check
			// If valued node go deeper else return no occlusion
			if(newNodeIndex == 0xFFFFFFFF) return 0;
			else nodeIndex = newNodeIndex + CalculateLevelChildId(voxPos, i + 1);
		}
		// Finally At requested level
		uint matLoc = offsetCascade.z + svoLevelOffset[depth - dimDepth.w] +
					  nodeIndex;
		return UnpackOcclusion(svoMaterial[matLoc].x);
	}
}

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per cone per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	uvec2 pixelId = globalId / uvec2(THREAD_PER_PIX, 1);
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
	float cascadeSpan = worldPos.Span.w * (0x1 << cascadeNo);

	// Find Edge vectors from normal
	// [(-z-y) / x, 1, 1] is perpendicular (unless normal is X axis)
	// handle special case where normal is (1.0f, 0.0f, 0.0f)
	vec3 ortho1 = normalize(vec3(-(worldNorm.z + worldNorm.y) / worldNorm.x, 1.0f, 1.0f));
	ortho1 = mix(ortho1, vec3(0.0f, 1.0f, 0.0f), floor(worldNorm.x));
	vec3 ortho2 = normalize(cross(worldNorm, ortho1));

	// Find Corner points of the surface
	float tanCone = tan(coneAngle);
	vec3 edgeMin = worldNorm - ortho1 * tanCone - ortho2 * tanCone;
	vec3 edgeMax = worldNorm + ortho1 * tanCone + ortho2 * tanCone;

	// Normally this surface is not flat but we consider it flat
	// For smaller angles this  should be true (considering voxels spanning a large area)



	// Start sampling towards that direction
	float totalConeOcclusion = 0.0f;
	float prevOccValue = 0.0f;
	for(float traversedDistance = 0.1f;	// Dont Start with zero, infinite loop
		traversedDistance <= maxDistance;)
	{
		// Determine Coverage Span of the surface 
		// (wrt cone angle and distance from pixel)
		


		uint nodeDepth = SpanToDepth(uint(round(diameter / worldPosSpan.w)));;
		float depthMultiplier =  0x1 << (dimDepth.y - nodeDepth);


		// Fetch Surface
		// Each Pixel needs to fetch 3x3 coverage points
		// 4 threads used for each pixel
		// 2 for each pixel + 1

	

		// Always fetch surfaces bigger than cone coverage
		// Surfaces should be flat for better sampling and consistency between pixels
		// Surfaces should be higher than the actual point in order to interpolate accrately

		// No need for sync threads (each thread in the same warp)
		
		// start sampling from that surface (interpolate)
		// need world space to surface space conversion
		// (point porjection)
		float interpolatedOcclusion;

		// than interpolate with your previous surface's value to simulate quadlinear interpolation
		float nodeOcclusion = mix(prevOcclusion,
		
		// do AO calculations from this value (or values)
		// Correction Term to prevent intersecting samples error
		nodeOcclusion = 1.0f - pow(1.0f - nodeOcclusion, marchDist / (depthMultiplier * worldPosSpan.w));
		
		// Occlusion falloff (linear)
		nodeOcclusion *= (1.0f / (1.0f + traversedDistance));//pow(traversedDistance, 0.5f))); 
		
		// Average total occlusion value
		totalConeOcclusion += (1 - totalConeOcclusion) * nodeOcclusion;


		// store the interpolated value as previous value

		// advance sample point (from sampling diameter)
		
	}


	surface [(BLOCK_SIZE_XY / 2) * SAMPLE_PER_PIXEL_XY][(BLOCK_SIZE_YY / 2) * SAMPLE_PER_PIXEL_XY];






	// Each Thread Has locally same location now generate cones
	// We will cast 4 Cones centered around the normal
	// we will choose two orthonormal vectors (wrt normal) in the plane defined by this normal and pos	
	// get and arbitrarty perpendicaular vector towards normal (N dot A = 0)
	// [(-z-y) / x, 1, 1] is one of those vectors (unless normal is X axis)
	vec3 ortho1 = normalize(vec3(-(worldNorm.z + worldNorm.y) / worldNorm.x, 1.0f, 1.0f));
	if(worldNorm.x == 1.0f) ortho1 = vec3(0.0f, 1.0f, 0.0f);
	vec3 ortho2 = normalize(cross(worldNorm, ortho1));

	// Determine your cone edge dir and your cone dir
	vec2 coneId = CONE_ID_MAP[globalId.x % CONE_COUNT];
	coneId = (coneId * 2.0f - 1.0f);
	vec2 coneIdEdge = coneId * tan(coneAngle);
	vec2 coneIdDir = coneId * tan(coneAngle * 0.5f);
	vec3 coneDir = worldNorm + ortho1 * coneId.x + ortho2 * coneId.y;
	coneDir = normalize(coneDir);
	float coneDiameterRatio = tan(coneAngle * 0.5f) * 2.0f;

	float gripSpanSize = worldPosSpan.w * (0x1 <<  cascadeNo);
	worldPos += coneDir * gripSpanSize;

	// Start sampling towards that direction
	float totalConeOcclusion = 0.0f;
	float traversedDistance = 0.1f;	// Dont Start with zero to make sample depth 0
	while(traversedDistance <= maxDistance)
	{
		// Calculate cone sphere diameter at the point
		vec3 coneRelativeLoc = coneDir * traversedDistance;
		float diameter = coneDiameterRatio * traversedDistance;

		// Select SVO Depth Relative to the current cone radius
		uint nodeDepth = SpanToDepth(uint(ceil(diameter / worldPosSpan.w)));
		nodeDepth = min(nodeDepth, dimDepth.y - cascadeNo);

		//DEBUG
		//nodeDepth = dimDepth.y;
		//nodeDepth = min(nodeDepth, dimDepth.y - cascadeNo);

		// SVO Query
		float nodeOcclusion = SampleSVOOcclusion(worldPos + coneRelativeLoc, nodeDepth);

		// Omit if %100 occuluded in closer ranges
		// Since its not always depth pos aligned with voxel pos
		bool isOmitDistance = (nodeOcclusion > 0.0f) &&
							  (traversedDistance < (SQRT3 * gripSpanSize));
		nodeOcclusion = isOmitDistance ? 0.0f : nodeOcclusion;		

		// March Distance
		float marchDist = diameter * sampleDistanceRatio;

		// Correction Term to prevent intersecting samples error
		float depthMultiplier =  0x1 << (dimDepth.y - nodeDepth);
		nodeOcclusion = 1.0f - pow(1.0f - nodeOcclusion, marchDist / (depthMultiplier * worldPosSpan.w));
		
		// Occlusion falloff (linear)
		nodeOcclusion *= (1.0f / (1.0f + traversedDistance));//pow(traversedDistance, 0.5f))); 
		
		// Average total occlusion value
		totalConeOcclusion += (1 - totalConeOcclusion) * nodeOcclusion;

		// Traverse Further
		traversedDistance += marchDist;
	}

	// Exchange Data Between cones (total is only on leader)
	// CosTetha multiplication
	totalConeOcclusion *= dot(worldNorm, coneDir);
	SumPixelOcclusion(totalConeOcclusion);

	totalConeOcclusion *= 0.25f;
	totalConeOcclusion *= 1.2f;


	// Logic Change (image write)
	if(globalId.x % CONE_COUNT == 0)
	{
		imageStore(liTex, ivec2(pixelId), vec4(vec3(1.0f - totalConeOcclusion), 0.0f));
		//imageStore(liTex, ivec2(pixelId), vec4(coneDir, 0.0f));
	}
		
}
