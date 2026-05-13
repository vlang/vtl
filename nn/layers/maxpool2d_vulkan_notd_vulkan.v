module layers

import vtl

pub struct MaxPool2DLayerVulkan[T] {
pub mut:
	kernel_size [2]int
	stride      [2]int
	padding     [2]int
	device      voidptr = unsafe { nil }
}

pub fn maxpool2d_forward_vulkan[T](input &vtl.Tensor[T], kernel_size [2]int, stride [2]int, padding [2]int, dev voidptr) !&vtl.Tensor[T] {
	return error('maxpool2d_forward_vulkan: Vulkan not enabled (compile with -d vulkan)')
}

pub fn (layer &MaxPool2DLayerVulkan[T]) forward(input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return error('MaxPool2DLayerVulkan: Vulkan not enabled (compile with -d vulkan)')
}
