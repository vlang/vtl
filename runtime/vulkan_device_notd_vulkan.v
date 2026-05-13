module runtime

// VulkanDevice is a stub type used when Vulkan is not enabled (-d vulkan not set).
// Calling new_vulkan_device() always returns an error in this path.
// Layer functions that accept VulkanDevice return errors without doing any work.
pub struct VulkanDevice {}

// new_vulkan_device returns an error because Vulkan is not enabled.
// Compile with -d vulkan to get a real device.
pub fn new_vulkan_device() !VulkanDevice {
	return error('Vulkan not enabled; compile with -d vulkan')
}

// release is a no-op in the non-Vulkan build.
pub fn (mut d VulkanDevice) release() ! {}

// device returns a nil placeholder in non-Vulkan builds.
pub fn (d &VulkanDevice) device() voidptr {
	return unsafe { nil }
}
