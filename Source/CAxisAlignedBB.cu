#include "CAxisAlignedBB.cuh"

__device__ bool IntersectsEBEN1992(const CAABB& boxA, const CAABB& boxB)
{
	//return ((boxA.max.x > boxB.min.x) && (boxB.max.x > boxA.min.x) &&
	//		(boxA.max.y > boxB.min.y) && (boxB.max.y > boxA.min.y) &&
	//		(boxA.max.z > boxB.min.z) && (boxB.max.z > boxA.min.z));

	if(boxA.min.x > boxB.max.x) return false;
	if(boxA.min.y > boxB.max.y) return false;
	if(boxA.min.z > boxB.max.z) return false;
	if(boxA.max.x < boxB.min.x) return false;
	if(boxA.max.y < boxB.min.y) return false;
	if(boxA.max.z < boxB.min.z) return false;
	return true;
}