#version 430
//#extension GL_NV_gpu_shader5 : require
//#extension GL_ARB_gpu_shader_int64 : require
//#extension GL_NV_shader_atomic_int64 : require
//#extension GL_NV_shader_atomic_float : require
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

#define U_SPAN layout(location = 1)
#define U_VOLUME_CORNER layout(location = 3)
#define U_GRID_SIZE layout(location = 4)

#define LU_VOXEL_RENDER layout(std430, binding = 1) coherent volatile
#define LU_ALLOCATOR layout(std430, binding = 2) coherent volatile

#define I_LOCK layout(r32ui, binding = 0) coherent volatile

#define T_COLOR layout(binding = 0)

// Input
in IN_UV vec2 fUV;
in IN_NORMAL vec3 fNormal;
in IN_POS vec3 fPos;

// Textures
uniform T_COLOR sampler2D colorTex;

// Images
uniform I_LOCK uimage3D lock;

// Uniforms
U_SPAN uniform float span;
U_GRID_SIZE uniform uvec3 gridSize;
U_VOLUME_CORNER uniform vec3 gridCorner;

LU_VOXEL_RENDER buffer VoxelData
{
	uvec2 voxelData[];
};

LU_ALLOCATOR buffer Allocator
{
	uint allocator;
};

uint UnpackCount(in uint packedData)
{
	return packedData >> 24;
}

vec3 UnpackNormal(in uint packedData)
{
	return unpackSnorm4x8(packedData).xyz;
}

vec4 UnpackAlbedo(in uint packedData)
{
	return unpackUnorm4x8(packedData);
}

uvec2 PackVoxel(in vec3 normal, in vec4 albedo, in uint count)
{
	uvec2 split;
	split.x = packUnorm4x8(albedo);
	
	//// Commented one does not work but 
	//// hand made one works i have no idea ....
	////uint nPack = packSnorm4x8(vec4(normal, 1.0f));
	//uint nPack = 0;
	//nPack |= uint(round(clamp(normal.z, -1.0f, 1.0f) * 127.0f)) << 16;
	//nPack |= uint(round(clamp(normal.y, -1.0f, 1.0f) * 127.0f)) << 8;
	//nPack |= uint(round(clamp(normal.x, -1.0f, 1.0f) * 127.0f)) << 0;

	split.y = 0;
	split.y |= ((count << 24) & 0xFF000000);
	//split.y |= (nPack & 0x00FFFFFF);
	return split;
}

uvec2 Average(in uvec2 avgIn, 
			  in vec3 normal, 
			  in vec4 albedo)
{
	// Unpack
	vec4 avgAlbedo = UnpackAlbedo(avgIn.x);
	vec3 avgNormal = UnpackNormal(avgIn.y);
	uint avgCount = UnpackCount(avgIn.y);
	
	// Average
	avgNormal = avgNormal * avgCount + normal;
	avgAlbedo = avgAlbedo * avgCount + albedo;
	avgCount = avgCount + 1;

	float countInv = 1.0f / float(avgCount);
	avgNormal *= countInv;
	avgAlbedo *= countInv;

	// Out
	return PackVoxel(avgNormal, avgAlbedo, avgCount);
}

void AtomicAverage(in vec3 normal, in vec4 albedo, in ivec3 iCoord)
{	
	uint coord = iCoord.z * gridSize.x * gridSize.y +
				 iCoord.y * gridSize.x +
			 	 iCoord.x;
//	// CAS Spin Lock
//	uint old;
//	do
//	{
//		old = imageAtomicCompSwap(lock, iCoord, 0, 1);
//		if(old == 0)
//		{
//			// Lock Acquired
//			// Fetch data
//			uvec2 data = voxelData[coord];
//			// Voxel counter increment if and only if you are the first
//			// node that is touches to this voxel
//			if(UnpackCount(data.y) == 0x0)
//			{
//				atomicAdd(allocator, 1);
//			}
//			voxelData[coord] = Average(data, normal, albedo);
//			
//			// Cache flush
//			memoryBarrier();
//			memoryBarrierBuffer();
//			
//			// Release Lock
//			imageAtomicExchange(lock, iCoord, 0);
//		}
//	}
//	while(old == 1);

	// Overwrite version
	voxelData[coord] = PackVoxel(normal, albedo, 1);

	//atomicAdd(allocator, 1);

	// Code graveyard below
	// uint64_t atomics crashes on glsl (CUDA atomics works fine)
	// I couldn't able to fix problem
	// I'll use 32-bit CAS lock instead

	//// This one crashes
	//// CAS Atomic
	//uint64_t assumed, old = voxelData[coord];
	//do
	//{
	//	assumed = old;
	//	//int64_t avg = Average(assumed, normal, albedo);
	//	uint64_t avg = assumed + 1;
	//	//old = atomicCompSwap(voxelData[coord], assumed, avg);
	//	old = atomicCompSwap(voxelData[coord], assumed, avg);
	//} while(1 != old);
	////while(false);

	////uint64_t assumed = allocator;
	//uint64_t assumed = voxelData[coord];
	//memoryBarrierBuffer();
	//bool loop = true;
	//while(loop)
	////for(int i = 0; i < 6000; i++)
	//{				
	//	uint64_t new = assumed + 1;
	//	//uint64_t new = Average(assumed, normal, albedo);
	//	//uint64_t old = atomicCompSwap(allocator, assumed, new);
	//	uint64_t old = atomicCompSwap(voxelData[coord], assumed, new);
	//	memoryBarrierBuffer();
	//	if(old == assumed)
	//	{
	//		//atomicAdd(allocator, 1);
	//		//i = 900000;
	//		loop = false;
	//	}
	//	//else i = 0;
	//	else assumed = old;
	//}

	//// This one crashes
	//// CAS Atomic
	//uint64_t assumed, old = voxelData[coord];
	//do
	//{
	//	assumed = old;
	//	//uint64_t avg = Average(assumed, normal, albedo);
	//	uint64_t avg = assumed + 1;
	//	old = atomicCompSwap(voxelData[coord], assumed, avg);
	//	i++;
	//} while(assumed != old);
	////while(false);

	//bool looping = true;
	//uint64_t assumed;
	//uint64_t old = voxelData[coord];
	//int i = 0;
	//while((i < 10) || looping)
	//{
	//	memoryBarrierBuffer();
	//	assumed = old;
	//	memoryBarrierBuffer();
	//	uint64_t new = assumed + 1;
	//	memoryBarrierBuffer();
	//	old = atomicCompSwap(voxelData[coord], assumed, new);
	//	memoryBarrierBuffer();
	//	if(old == assumed)
	//		looping = false;
	//	memoryBarrierBuffer();
	//	i++;
	//}

	//uint64_t assumed = 0x0;
	//uint64_t new = 1;
	//uint64_t old = atomicCompSwap(voxelData[0], assumed, new);
	//while(assumed != old)
	//{

	//}

	// CAS Atomic
	//bool loop = true;
	//uint64_t assumed = atomicAdd(voxelData[coord], 0);
	//memoryBarrierBuffer();
	//while(loop)
	//{
	//	uint64_t avg = assumed + 1;
	//	uint64_t old = atomicCompSwap(voxelData[coord], assumed, avg);
	//	memoryBarrierBuffer();
	//	if(old == assumed) loop = false;

	//	assumed = old;
	//}

	//// Comp swap generic atomic loop (not works)
	//uint64_t old, assumed;
	//old = voxelData[0];
	//do
	//{
	////	//assumed = old;

	////	//uint64_t new = assumed + 1;
	//	uint64_t new = 1;
	//	assumed = 0;

	//	old = atomicCompSwap(voxelData[0], assumed, new);

	////	memoryBarrierBuffer();
	////	memoryBarrier();

	//	if(old == assumed)
	//	{
	//		atomicAdd(allocator, 1);
	//		//allocator += 1;
	//		atomicExchange(voxelData[coord], assumed);
	//		memoryBarrierBuffer();
	//		memoryBarrier();
	//	}
	//} while(old != assumed);

	// -------------------------------------------------------------
	// Classic spin lock (This works)
	//uint64_t old;
	//do
	//{
	//	old = atomicCompSwap(voxelData[0], 0, 1);
	//	if(old == 0)
	//	{
	//		allocator += 1;
	//		atomicExchange(voxelData[0], 0);
	//		memoryBarrierBuffer();
	//	}
		
	//	if(old > 1)
	//	{
	//		atomicExchange(voxelData[coord], 5);
	//		memoryBarrierBuffer();
	//	}
		
	////} while(old != 0);
	//} while(old >= 1);
	//memoryBarrierBuffer();

	//atomicAdd(voxelData[coord], 1);
	//atomicAdd(allocator, 1);

	//{
	//	voxelData[coord] = 3;
	//	memoryBarrierBuffer();
	//		return;
	//	}
	//memoryBarrierBuffer();
	//memoryBarrier();
	//atomicAdd(allocator, 1);
	////voxelData[coord] = old;
	////voxelData[coord] = 3;
	//atomicAdd(allocator, 1);
	//memoryBarrierBuffer();
	

	// -------------------------------------------------------------

	//memoryBarrierBuffer();
	//uint64_ old = atomicAdd(voxelData[coord], 0);
	//uint64_t assumed;
	//memoryBarrierBuffer();
	//do
	//{
	//	memoryBarrierBuffer();
	//	assumed = old;
	//	memoryBarrierBuffer();
	//	//uint64_t avg = Average(assumed, normal, albedo);
	//	uint64_t avg = assumed + 1;
	//	memoryBarrierBuffer();
	//	old = atomicCompSwap(voxelData[coord], assumed, avg);
	//	memoryBarrierBuffer();
	//} while(assumed != old);

	//memoryBarrierBuffer();

	// Add that this node is valid
	//if(old == 0x0) atomicAdd(allocator, uint(old));
	//if(assumed == 0x0) atomicAdd(allocator, 1);

	//atomicAdd(allocator, uint(old));

	// Non atomic overwrite version
	//voxelData[coord] = PackVoxel(normal, albedo, 1); 

	// Old version that used 8 floats (so much memory)
	//// Thanks nvidia Kappa
	//atomicAdd(normalDense[coord].x, normal.x);
	//atomicAdd(normalDense[coord].y, normal.y);
	//atomicAdd(normalDense[coord].z, normal.z);
	//atomicAdd(normalDense[coord].w, 1.0f);
	
	//atomicAdd(albedoDense[coord].x, color.x);
	//atomicAdd(albedoDense[coord].y, color.y);
	//atomicAdd(albedoDense[coord].z, color.z);
	//atomicAdd(albedoDense[coord].w, specular);
}


//DEBUG
out layout(location = 0) vec4 testOut;

void main(void)
{
	// Find Index of Voxel
	vec3 relativePos = fPos - gridCorner;
	ivec3 index = ivec3(relativePos / span);

	// Material Fetch
	vec4 color = texture(colorTex, fUV).rgba;

	//// If in range average
	//if(index.x < gridSize.x &&
	//   index.y < gridSize.y &&
	//   index.z < gridSize.z &&
	//   index.x >= 0 &&
	//   index.y >= 0 &&
	//   index.z >= 0)
	//{
		AtomicAverage(fNormal, color, index);

		//atomicAdd(allocator, 1);

		// DEBUG
		testOut = vec4(color.rgb, 1.0f);
	//}
}