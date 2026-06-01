module layers

import vtl

pub fn relu_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	out := relu_forward_vulkan_f32(x) or { return none }
	return out
}

pub fn sigmoid_forward_f32_try(x &vtl.Tensor[f32]) ?&vtl.Tensor[f32] {
	out := sigmoid_forward_vulkan_f32(x) or { return none }
	return out
}
