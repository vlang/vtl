module layers

import vtl

pub fn relu_forward_vulkan_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	return error(@FN + ': compile with -d vulkan')
}

pub fn sigmoid_forward_vulkan_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	return error(@FN + ': compile with -d vulkan')
}
