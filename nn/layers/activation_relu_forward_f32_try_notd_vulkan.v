module layers

import vtl

// relu_forward_f32_try exposes this operation as part of the public API.
pub fn relu_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	_ = x
	return none
}

// sigmoid_forward_f32_try exposes this operation as part of the public API.
pub fn sigmoid_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	_ = x
	return none
}

// softplus_forward_f32_try exposes the Vulkan Softplus kernel when available.
pub fn softplus_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	_ = x
	return none
}

// selu_forward_f32_try exposes the Vulkan SELU kernel when available.
pub fn selu_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	_ = x
	return none
}

// hardswish_forward_f32_try exposes the Vulkan HardSwish kernel when available.
pub fn hardswish_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	_ = x
	return none
}
