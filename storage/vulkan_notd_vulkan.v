module storage

// VulkanParams defines a public data structure for this module.

// VulkanParams defines a public data structure for this module.
@[params]
pub struct VulkanParams {}

// vulkan exposes this operation as part of the public API.
pub fn (cpu &CpuStorage[T]) vulkan(params VulkanParams) !&CpuStorage[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vulkan" to use the Vulkan Backend')
}
