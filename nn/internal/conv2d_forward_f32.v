module internal

import vtl

// conv2d_forward_f32 uses Vulkan im2col+GEMM when eligible (`VTL_USE_VULKAN=1`, `-d vulkan`), else CPU.
pub fn conv2d_forward_f32(input &vtl.Tensor[f32],
	weight &vtl.Tensor[f32],
	bias &vtl.Tensor[f32],
	kernel_size []int,
	config Conv2DConfig) !&vtl.Tensor[f32] {
	if conv2d_vulkan_eligible(kernel_size, config) {
		if vk_out := conv2d_forward_vulkan_f32(input, weight, bias, kernel_size, config) {
			return vk_out
		}
	}
	return conv2d_forward_cpu_f32(input, weight, bias, kernel_size, config)
}
