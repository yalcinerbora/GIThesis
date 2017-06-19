#include <cuda.h>
#include <cuda_fp16.h>
#include "CSVOFunctions.cuh"

enum class CSimilarity
{
	RIGHT_CONBINABLE,	// These two can combine, right node is the neigbour of base node
	LEFT_CONBINABLE,	// These two can combine, left node is the neigbour of base node
	RIGHT_SUBSET,		// Right node covers left node
	LEFT_SUBSET,		// Left node covers right node
	EXACT,				// Exact match
	NONE				// No match
};

inline __device__ CSimilarity IsSimilar(CSVONode left, CSVONode right)
{
	// 
}

// Warp Level Reduction of redundant nodes
// Each Lane will find similar lanes and one of those lanes will return
// node using outNode variable
// variable returns true if this lane required to add a node to the shared memory hash
// false otherwise
inline __device__ bool ReducePeers(CSVONode& outNode, CSVONode node)
{
	//uint peers = 0;
	//bool is_peer;

	//// in the beginning, all lanes are available
	//uint unclaimed = 0xffffffff;

	//do
	//{
	//	// fetch key of first unclaimed lane and compare with this key
	//	is_peer = (key == __shfl(key, __ffs(unclaimed) - 1));

	//	// determine which lanes had a match
	//	peers = __ballot(is_peer);

	//	// remove lanes with matching keys from the pool
	//	unclaimed ^= peers;

	//	// quit if we had a match
	//} while(!is_peer);

	//return peers;
}
