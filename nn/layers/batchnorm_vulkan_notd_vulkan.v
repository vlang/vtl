module layers

import vtl

pub struct BatchNorm1DLayerVulkan[T] {
	eps    f32
	device voidptr = unsafe { nil }
}

pub fn batchnorm1d_forward_vulkan[T](input &vtl.Tensor[T], eps f32, dev voidptr) !&vtl.Tensor[T] {
	return error('batchnorm1d_forward_vulkan: Vulkan not enabled (compile with -d vulkan)')
}

pub fn (layer &BatchNorm1DLayerVulkan[T]) forward(input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return error('BatchNorm1DLayerVulkan: Vulkan not enabled (compile with -d vulkan)')
}
