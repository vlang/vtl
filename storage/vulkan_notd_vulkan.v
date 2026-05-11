module storage

@[params]
pub struct VulkanStorageParams {
pub:
	device voidptr = unsafe { nil }
}

pub fn new_vulkan_params(dev voidptr) VulkanStorageParams {
	return VulkanStorageParams{ device: dev }
}
