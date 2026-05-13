module runtime

import vsl.vulkan

// VulkanDevice wraps a Vulkan compute device for use in VTL tests and layers.
// Acquire one via runtime.new_vulkan_device().
// This type is only defined when compiling with -d vulkan.
pub struct VulkanDevice {
pub mut:
	inner &vulkan.Device
}

// new_vulkan_device creates a VulkanDevice from the default Vulkan compute device.
// Returns an error if no Vulkan-capable device is found on the system.
pub fn new_vulkan_device() !VulkanDevice {
	dev := vulkan.new_device()!
	return VulkanDevice{
		inner: dev
	}
}

// release destroys the underlying Vulkan device and frees its resources.
pub fn (mut d VulkanDevice) release() ! {
	d.inner.release()!
}

// device returns the raw *vulkan.Device pointer.
// Only call this from _d_vulkan.v layer implementations or inside $if vulkan ? {} blocks.
pub fn (d &VulkanDevice) device() &vulkan.Device {
	return d.inner
}
