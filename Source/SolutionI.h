/**

Solution implementtion

*/

#ifndef __SOLUTIONI_H__
#define __SOLUTIONI_H__

class SceneI;
struct Camera;

class SolutionI
{
	public:
		//virtual			~SolutionI() = default;

		// Interface
		virtual bool	IsCurrentScene(SceneI&) = 0;
		virtual void	Init(SceneI&) = 0;
		virtual void	Release() = 0;
		virtual void	Frame(const Camera&) = 0;
		virtual void	SetFPS(double fpsMS) = 0;
};
#endif //__SOLUTIONI_H__