module layers

import vtl

// VulkanParams defines a public data structure for this module.

// VulkanParams defines a public data structure for this module.
@[params]
pub struct VulkanParams {}

// linear_forward_vulkan_f32 stub when not built with `-d vulkan`.
pub fn linear_forward_vulkan_f32(x &vtl.Tensor[f32], weights &vtl.Tensor[f32], bias &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	return error(@FN + ': compile with -d vulkan to use Vulkan linear forward')
}

// linear_forward_vulkan returns error if Vulkan is not enabled
pub fn linear_forward_vulkan[T](x &Tensor[T], weights &Tensor[T], bias &Tensor[T]) !&Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vulkan" to use the Vulkan Backend')
}

// relu_forward_vulkan returns error if Vulkan is not enabled
pub fn relu_forward_vulkan[T](x &Tensor[T]) !&Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vulkan" to use the Vulkan Backend')
}

// sigmoid_forward_vulkan returns error if Vulkan is not enabled
pub fn sigmoid_forward_vulkan[T](x &Tensor[T]) !&Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vulkan" to use the Vulkan Backend')
}
