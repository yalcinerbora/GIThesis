/**

*/

#ifndef __SCENEI_H__
#define __SCENEI_H__

class SceneI 
{
	private:
	protected:
	public:
		virtual					~SceneI() = default;

		// Interface
		virtual void			Draw() = 0;
};

#endif //__SCENEI_H__