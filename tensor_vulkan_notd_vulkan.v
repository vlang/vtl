module vtl

import storage

// Stub when compiling without `-d vulkan` (mirrors tensor_vcl_notd_vcl.v).
pub fn (t &Tensor[T]) vulkan(params storage.VulkanStorageParams) !&Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vulkan" to use the Vulkan backend')
}
