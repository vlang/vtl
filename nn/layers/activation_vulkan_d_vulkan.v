module layers

import vtl
import vtl.storage
import vsl.vulkan
import vsl.vulkan.compute

fn activation_vulkan_device() !&vulkan.Device {
	mut probe := vtl.zeros[f32]([1])
	vk := probe.vulkan(storage.VulkanParams{})!
	defer { vk.release() }
	return vk.data.data.device
}

// relu_forward_vulkan_f32 uses VSL unary GPU ops (host I/O) — avoids VulkanTensor method codegen issues.
pub fn relu_forward_vulkan_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	dev := activation_vulkan_device()!
	out_arr := compute.relu_vulkan_f32(dev, x.to_array())!
	return vtl.from_array(out_arr, x.shape)!
}

// sigmoid_forward_vulkan_f32 uses VSL unary GPU ops (host I/O).
pub fn sigmoid_forward_vulkan_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	dev := activation_vulkan_device()!
	out_arr := compute.sigmoid_vulkan_f32(dev, x.to_array())!
	return vtl.from_array(out_arr, x.shape)!
}
