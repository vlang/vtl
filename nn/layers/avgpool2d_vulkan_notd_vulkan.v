module layers

import vtl

// AvgPool2DLayerVulkan stub for when Vulkan is not enabled
pub struct AvgPool2DLayerVulkan[T] {
pub mut:
	kernel_size [2]int
	stride      [2]int
	padding     [2]int
	device      voidptr
}

// avgpool2d_forward_vulkan stub
pub fn avgpool2d_forward_vulkan[T](input &vtl.Tensor[T], kernel_size [2]int, stride [2]int, padding [2]int, dev voidptr) !&vtl.Tensor[T] {
	return error('AvgPool2D Vulkan not enabled; compile with -d vulkan')
}

// forward stub
pub fn (layer &AvgPool2DLayerVulkan[T]) forward(input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return error('AvgPool2D Vulkan not enabled; compile with -d vulkan')
}
