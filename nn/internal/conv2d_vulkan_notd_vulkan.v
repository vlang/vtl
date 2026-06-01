module internal

import vtl

// conv2d_vulkan_eligible is false without `-d vulkan`.
pub fn conv2d_vulkan_eligible(kernel_size []int, config Conv2DConfig) bool {
	return false
}

// conv2d_forward_vulkan_f32 stub when not built with `-d vulkan`.
pub fn conv2d_forward_vulkan_f32(input &vtl.Tensor[f32],
	weight &vtl.Tensor[f32],
	bias &vtl.Tensor[f32],
	kernel_size []int,
	config Conv2DConfig) !&vtl.Tensor[f32] {
	return error(@FN + ': compile with -d vulkan for Vulkan conv2d')
}
