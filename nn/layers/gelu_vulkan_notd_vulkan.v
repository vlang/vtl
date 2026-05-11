module layers

import vtl
import storage

pub fn gelu_forward_vulkan[T](x &vtl.Tensor[T], params storage.VulkanStorageParams) !&vtl.Tensor[T] {
	return error('gelu_forward_vulkan: Vulkan not enabled (compile with -d vulkan)')
}
