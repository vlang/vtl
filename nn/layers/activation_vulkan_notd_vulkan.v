module layers

import vtl

// relu_forward_vulkan_f32 exposes this operation as part of the public API.
pub fn relu_forward_vulkan_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	return error(@FN + ': compile with -d vulkan')
}

// sigmoid_forward_vulkan_f32 exposes this operation as part of the public API.
pub fn sigmoid_forward_vulkan_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	return error(@FN + ': compile with -d vulkan')
}
