module internal

import vtl

pub fn linear_backward_vulkan_f32(grad &vtl.Tensor[f32], input &vtl.Tensor[f32],
	weight &vtl.Tensor[f32], bias_needs_grad bool) ![]&vtl.Tensor[f32] {
	_ = grad
	_ = input
	_ = weight
	_ = bias_needs_grad
	return error(@FN + ': compile with -d vulkan')
}
