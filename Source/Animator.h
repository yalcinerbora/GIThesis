/**

*/

#ifndef __ANIMATOR_H__
#define __ANIMATOR_H__

#include "Shader.h"

class MeshBatchSkeletal;

class Animator
{
	private:
		Shader				compGenInvBind;
		Shader				compInterpAnim;
		Shader				compAnimate;

	protected:
	public:
							Animator();

		// 
		void				Update(MeshBatchSkeletal&) const;

};

#endif //__ANIMATOR_H__