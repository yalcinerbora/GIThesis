#include "Animator.h"

Animator::Animator()
	: compGenInvBind(ShaderType::COMPUTE, "Shaders/GenInvBindPose.glsl")
	, compInterpAnim(ShaderType::COMPUTE, "Shaders/InterpAnimation.glsl")
	, compAnimate(ShaderType::COMPUTE, "Shaders/Animate.glsl")
{}


void Animator::Update(MeshBatchSkeletal& s) const
{

}