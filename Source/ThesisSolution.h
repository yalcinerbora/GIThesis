/**

Solution implementtion

*/

#ifndef __THESISSOLUTION_H__
#define __THESISSOLUTION_H__

#include <vector>
#include "SolutionI.h"
#include "Shader.h"
#include "FrameTransformBuffer.h"
#include "GICudaVoxelScene.h"

class ThesisSolution : public SolutionI
{
	private:
		SceneI*					currentScene;

		Shader					vertexDebugVoxel;
		Shader					fragmentDebugVoxel;

		Shader					vertexVoxelizeObject;
		Shader					fragmentVoxelizeObject;
		Shader					fragmentVoxelizeCount;
		Shader					computeDetermineVoxSpan;

		FrameTransformBuffer	cameraTransform;

		std::vector<GLuint>		voxelObjectData;
		std::vector<GLuint>		voxelObjectRenderData;
		std::vector<GLuint>		voxelVAO;

		std::vector<size_t>		objVoxelSizes;
		GLuint					objGridInfoBuffer;

		// Cuda Segment
		GICudaVoxelScene		voxelScene;

	protected:
		void					CreateVoxel(size_t index);
		void					DrawVoxel(size_t index);

	public:
								ThesisSolution();
								~ThesisSolution() = default;

		// Interface
		bool					IsCurrentScene(SceneI&) override;
		void					Init(SceneI&) override;
		void					Frame(const Camera&) override;
};
#endif //__THESISSOLUTION_H__