module optimizers

import vtl
import vtl.autograd

// adam_use_vulkan_optimizer exposes this operation as part of the public API.
pub fn adam_use_vulkan_optimizer() bool {
	return false
}

// try_adam_update_f32_vulkan exposes this operation as part of the public API.
pub fn try_adam_update_f32_vulkan(mut v autograd.Variable[f32], mut m_tensor vtl.Tensor[f32],
	mut v_tensor vtl.Tensor[f32], step AdamStepParams) bool {
	return false
}
