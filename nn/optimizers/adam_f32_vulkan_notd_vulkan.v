module optimizers

import vtl
import vtl.autograd

pub fn adam_use_vulkan_optimizer() bool {
	return false
}

pub fn try_adam_update_f32_vulkan(mut v autograd.Variable[f32], mut m_tensor vtl.Tensor[f32],
	mut v_tensor vtl.Tensor[f32], step AdamStepParams) bool {
	return false
}
