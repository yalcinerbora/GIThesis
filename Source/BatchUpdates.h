/**


*/
#ifndef __BATCH_UPDATES_H__
#define __BATCH_UPDATES_H__

class DrawBuffer;
class GPUBuffer;

namespace BatchUpdates
{
	void SponzaUpdate(GPUBuffer&, DrawBuffer&, double);
	void CornellUpdate(GPUBuffer&, DrawBuffer&, double);
	void CubeUpdate(GPUBuffer&, DrawBuffer&, double);
}
#endif //__BATCH_UPDATES_H__