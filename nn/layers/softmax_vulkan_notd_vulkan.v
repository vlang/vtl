module layers

// Stub implementations for non-Vulkan builds.
// The real implementations live in softmax_vulkan_d_vulkan.v

import vtl
import storage

pub fn softmax_forward_vulkan[T](x &vtl.Tensor[T], params storage.VulkanStorageParams) !&vtl.Tensor[T] {
	return error('softmax_forward_vulkan: Vulkan not enabled (compile with -d vulkan)')
}

pub fn layernorm_forward_vulkan[T](x &vtl.Tensor[T], eps f32, params storage.VulkanStorageParams) !&vtl.Tensor[T] {
	return error('layernorm_forward_vulkan: Vulkan not enabled (compile with -d vulkan)')
}

pub fn reduce_sum_vulkan[T](x &vtl.Tensor[T], params storage.VulkanStorageParams) ![]T {
	return error('reduce_sum_vulkan: Vulkan not enabled (compile with -d vulkan)')
}
