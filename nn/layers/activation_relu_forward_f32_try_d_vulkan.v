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
