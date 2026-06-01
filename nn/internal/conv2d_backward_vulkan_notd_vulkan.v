module internal

import vtl

// conv2d_backward_vulkan_f32 exposes this operation as part of the public API.
pub fn conv2d_backward_vulkan_f32(grad_out &vtl.Tensor[f32], input &vtl.Tensor[f32], weight &vtl.Tensor[f32],
	bias &vtl.Tensor[f32], kernel_size []int, config Conv2DConfig) ![]&vtl.Tensor[f32] {
	return error(@FN + ': compile with -d vulkan for Vulkan conv2d backward')
}
