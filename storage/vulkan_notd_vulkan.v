module storage

@[params]
pub struct VulkanParams {}

pub fn (cpu &CpuStorage[T]) vulkan(params VulkanParams) !&CpuStorage[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vulkan" to use the Vulkan Backend')
}
