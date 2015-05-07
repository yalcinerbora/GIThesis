/**

*/

#ifndef __SCENEI_H__
#define __SCENEI_H__

class DrawBuffer;

class SceneI 
{
	private:
	protected:
	public:
		virtual					~SceneI() = default;

		// Interface
		virtual void			Draw() = 0;
		virtual DrawBuffer&		getDrawBuffer() = 0;

		virtual size_t			ObjectCount() const = 0;
};

#endif //__SCENEI_H__