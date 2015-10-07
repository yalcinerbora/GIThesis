#version 430
				
// Definitions
#define LU_VOXEL_NORM_POS layout(std430, binding = 0) restrict
#define LU_VOXEL_RENDER layout(std430, binding = 1) restrict 
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2) restrict readonly
#define LU_VOXEL_IDS layout(std430, binding = 4) restrict
#define LU_INDEX_CHECK layout(std430, binding = 5) restrict
#define LU_VOXEL_NORM_POS_OBJ layout(std430, binding = 6) restrict readonly
#define LU_VOXEL_IDS_OBJ layout(std430, binding = 7) restrict readonly
#define LU_VOXEL_RENDER_OBJ layout(std430, binding = 8) restrict readonly
#define LU_ATOMIC_COUNTER layout(std430, binding = 9) restrict 
#define LU_SORT_INDICES layout(std430, binding = 10) restrict readonly

#define U_OBJ_ID layout(location = 4)
#define U_MAX_CACHE_SIZE layout(location = 5)

// I-O
U_MAX_CACHE_SIZE uniform uint maxSize;
U_OBJ_ID uniform uint objId;

LU_INDEX_CHECK buffer CountArray
{
	uint writeIndex;
};

LU_ATOMIC_COUNTER buffer LocalSizeArray
{
	uint localSize;
};

LU_OBJECT_GRID_INFO buffer GridInfo
{
	struct
	{
		float span;
		uint voxCount;
	} objectGridInfo[];
};

LU_VOXEL_NORM_POS buffer VoxelNormPosArray
{
	uvec2 voxelNormPos[];
};

LU_VOXEL_IDS buffer VoxelIdsArray
{
	uvec2 voxelIds[];
};

LU_SORT_INDICES buffer SortIndices
{
	unsigned int sortIndex[];
};

LU_VOXEL_RENDER buffer VoxelArrayRender
{
	uint voxelArrayRender[];
};

LU_VOXEL_NORM_POS_OBJ buffer NormPosObj
{
	uvec2 objNormPos[];
};

LU_VOXEL_IDS_OBJ buffer IdsObj
{
	uvec2 objIds[];
};

LU_VOXEL_RENDER_OBJ buffer RenderObj
{
	uint objRender[];
};

uvec2 InjectRenderIndex(in uvec2 objId, in uint voxId)
{
	return uvec2(objId.x, voxId);
}

uvec2 InjectNormal(in uvec2 normPos, in vec3 normal)
{
	uvec2 result = uvec2(0);

	// 1615 XY Format
	// 32 bit format LS 16 bits are X
	// MSB is the sign of Z
	// Rest is Y
	// both x and y is SNORM types
	uvec2 value = uvec2(0.0f);
	value.x = uint((normal.x * 0.5f + 0.5f) * 0xFFFF);
	value.y = uint((normal.y * 0.5f + 0.5f) * 0x7FFF);
	value.y |= (floatBitsToUint(normal.z) >> 16) & 0x00008000;
	
	result.y = value.y << 16;
	result.y |= value.x;

	result.x = normPos.x;
	return result;
}

vec3 ExpandNormal(in uint norm)
{
	vec3 result;
	result.x = ((float(norm & 0x0000FFFF) / 0xFFFF) - 0.5f) * 2.0f;
	result.y = ((float((norm & 0x7FFF0000) >> 16) / 0x7FFF) - 0.5f) * 2.0f;
	result.z = sqrt(abs(1.0f - dot(result.xy, result.xy)));
	result.z *= sign(int(norm & 0x80000000));
	return result;
}

uvec3 ExpandColor(in uint color)
{
	uvec3 result = uvec3(0);

	result.x = color &  0x000000FF;
	result.y = (color & 0x0000FF00) >> 8;
	result.z = (color & 0x00FF0000) >> 16;
	return result;
}

uint PackColor(in uvec3 color)
{
	uint colorPacked = 0;
	colorPacked = color.r & 0x000000FF;
	colorPacked |= color.g & 0x000000FF << 8;
	colorPacked |= color.b & 0x000000FF << 16;
	return colorPacked;
}

layout (local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
void main(void)
{
	uint voxId = gl_GlobalInvocationID.x;
	if(voxId >= objectGridInfo[objId].voxCount) return;
	
	//if(voxId == 0 ||
	//	(voxId != 0 && objNormPos[voxId] != objNormPos[voxId - 1]))
	//{
		uint index = atomicAdd(writeIndex, 1);
		if(index >= maxSize) return;

		atomicAdd(localSize, 1);

		//// This hurts to write...
		//// Branch divergence and global accessing for loop
		//// Still its the easiest to implement
		//// And here is not performance critical
		//// Average Values
		//uvec3 color = uvec3(0);
		//vec3 normal = vec3(0);
		//int i = -1;
		//do
		//{
		//	i++;
		//	normal += ExpandNormal(objNormPos[voxId + i].y);
		//	color += ExpandColor(objRender[voxId + i]);
		//	normalize(normal);		
		//} while (voxId + i >= objectGridInfo[objId].voxCount || 
		//		(voxId + i >= objectGridInfo[objId].voxCount &&
		//		objNormPos[voxId + i + 1] != objNormPos[voxId + i + 2]));
		//color /= i;

		// Transfer to Object Batch
		//voxelNormPos[index] = InjectNormal(objNormPos[voxId], normal);
		//voxelIds[index] = InjectRenderIndex(objIds[voxId], index);
		//voxelArrayRender[index] = PackColor(color);

		voxelNormPos[index] = objNormPos[sortIndex[voxId]];
		voxelIds[index] = objIds[sortIndex[voxId]];//InjectRenderIndex(objIds[voxId], index);
		voxelArrayRender[index] = objRender[sortIndex[voxId]];
	//}	
}