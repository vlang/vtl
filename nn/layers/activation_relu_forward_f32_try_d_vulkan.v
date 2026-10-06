module layers

import vtl

// relu_forward_f32_try exposes this operation as part of the public API.
pub fn relu_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	out := relu_forward_vulkan_f32(x) or { return none }
	return out
}

// sigmoid_forward_f32_try exposes this operation as part of the public API.
pub fn sigmoid_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	out := sigmoid_forward_vulkan_f32(x) or { return none }
	return out
}

// softplus_forward_f32_try exposes the Vulkan Softplus kernel when available.
pub fn softplus_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	out := softplus_forward_vulkan_f32(x) or { return none }
	return out
}

// selu_forward_f32_try exposes the Vulkan SELU kernel when available.
pub fn selu_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	out := selu_forward_vulkan_f32(x) or { return none }
	return out
}

// hardswish_forward_f32_try exposes the Vulkan HardSwish kernel when available.
pub fn hardswish_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	out := hardswish_forward_vulkan_f32(x) or { return none }
	return out
}
