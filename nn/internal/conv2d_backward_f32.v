module internal

import vtl

// conv2d_backward_f32 uses Vulkan GEMM for d_weight when eligible, else full CPU.
pub fn conv2d_backward_f32(grad_out &vtl.Tensor[f32], input &vtl.Tensor[f32], weight &vtl.Tensor[f32],
	bias &vtl.Tensor[f32], kernel_size []int, config Conv2DConfig) ![]&vtl.Tensor[f32] {
	if conv2d_vulkan_backward_eligible(kernel_size, config) {
		if vk := conv2d_backward_vulkan_f32(grad_out, input, weight, bias, kernel_size, config) {
			return vk
		}
	}
	return conv2d_backward_cpu_f32(grad_out, input, weight, bias, kernel_size, config)
}
