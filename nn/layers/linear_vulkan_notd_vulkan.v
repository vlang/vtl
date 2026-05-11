module layers

import vtl
import vtl.storage

// Stubs when compiling without `-d vulkan` (mirrors nn + tensor notd patterns).

pub fn linear_forward_vulkan[T](_ &vtl.Tensor[T], _ &vtl.Tensor[T], _ &vtl.Tensor[T], _ storage.VulkanStorageParams) !&vtl.Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vulkan" to use Vulkan forward paths')
}

pub fn relu_forward_vulkan[T](_ &vtl.Tensor[T], _ storage.VulkanStorageParams) !&vtl.Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vulkan" to use Vulkan forward paths')
}

pub fn sigmoid_forward_vulkan[T](_ &vtl.Tensor[T], _ storage.VulkanStorageParams) !&vtl.Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vulkan" to use Vulkan forward paths')
}
