module layers

import vtl

// GlobalAvgPool2DLayerVulkan stub for when Vulkan is not enabled
pub struct GlobalAvgPool2DLayerVulkan[T] {
pub mut:
	device voidptr
}

// global_avgpool2d_forward_vulkan stub
pub fn global_avgpool2d_forward_vulkan[T](input &vtl.Tensor[T], dev voidptr) !&vtl.Tensor[T] {
	return error('GlobalAvgPool2D Vulkan not enabled; compile with -d vulkan')
}

// forward stub
pub fn (layer &GlobalAvgPool2DLayerVulkan[T]) forward(input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return error('GlobalAvgPool2D Vulkan not enabled; compile with -d vulkan')
}
