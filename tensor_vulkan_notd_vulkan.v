module vtl

import vtl.storage
import vsl.vulkan as _

pub fn (t &Tensor[T]) vulkan(params storage.VulkanParams) !&Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vulkan" to use the Vulkan Backend')
}
