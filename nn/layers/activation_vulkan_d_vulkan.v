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

// softplus_forward_vulkan_f32 applies Softplus using the VSL Vulkan kernel.
pub fn softplus_forward_vulkan_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	dev := activation_vulkan_device()!
	out_arr := compute.softplus_vulkan_f32(dev, x.to_array())!
	return vtl.from_array(out_arr, x.shape)!
}

// selu_forward_vulkan_f32 applies SELU using the VSL Vulkan kernel.
pub fn selu_forward_vulkan_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	dev := activation_vulkan_device()!
	out_arr := compute.selu_vulkan_f32(dev, x.to_array())!
	return vtl.from_array(out_arr, x.shape)!
}

// hardswish_forward_vulkan_f32 applies HardSwish using the VSL Vulkan kernel.
pub fn hardswish_forward_vulkan_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	dev := activation_vulkan_device()!
	out_arr := compute.hardswish_vulkan_f32(dev, x.to_array())!
	return vtl.from_array(out_arr, x.shape)!
}
