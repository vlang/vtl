module optimizers

import os
import vtl
import vtl.autograd
import vtl.storage
import vsl.vulkan
import vsl.vulkan.compute

pub fn adam_use_vulkan_optimizer() bool {
	return os.getenv('VTL_USE_VULKAN') == '1'
}

fn adam_vulkan_device() !&vulkan.Device {
	mut probe := vtl.zeros[f32]([1])
	vk := probe.vulkan(storage.VulkanParams{})!
	defer { vk.release() }
	return vk.data.data.device
}

// try_adam_update_f32_vulkan returns true when GPU Adam succeeded.
pub fn try_adam_update_f32_vulkan(mut v autograd.Variable[f32], mut m_tensor vtl.Tensor[f32],
	mut v_tensor vtl.Tensor[f32], step AdamStepParams) bool {
	adam_update_f32_vulkan(mut v, mut m_tensor, mut v_tensor, step) or { return false }
	return true
}

// adam_update_f32_vulkan runs fused Adam on GPU (VSL adam_step shader).
fn adam_update_f32_vulkan(mut v autograd.Variable[f32], mut m_tensor vtl.Tensor[f32],
	mut v_tensor vtl.Tensor[f32], step AdamStepParams) ! {
	dev := adam_vulkan_device()!
	grad := v.grad.to_array()
	mut theta := v.value.to_array()
	mut m_arr := m_tensor.to_array()
	mut v_arr := v_tensor.to_array()
	compute.adam_step_vulkan_f32(dev, grad, mut theta, mut m_arr, mut v_arr, compute.AdamStepParams{
		beta1:   step.beta1
		beta2:   step.beta2
		lr_t:    step.lr_t
		epsilon: step.epsilon
	})!
	v.value = vtl.from_array(theta, v.value.shape)!
	m_tensor = vtl.from_array(m_arr, v.value.shape)!
	v_tensor = vtl.from_array(v_arr, v.value.shape)!
}
