module models

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.storage
import vsl.vulkan

struct VulkanInferenceSession {
mut:
	device   &vulkan.Device       = unsafe { nil }
	resident &vtl.VulkanTensor[f32] = unsafe { nil }
}

fn (mut session VulkanInferenceSession) ensure_device() ! {
	if session.device == unsafe { nil } {
		session.device = vulkan.new_device()!
	}
}

fn (mut session VulkanInferenceSession) release_resident() {
	if session.resident != unsafe { nil } {
		session.resident.release()
		session.resident = unsafe { nil }
	}
}

fn (mut session VulkanInferenceSession) release() ! {
	session.release_resident()
	if session.device != unsafe { nil } {
		session.device.release()!
		session.device = unsafe { nil }
	}
}

// forward_vulkan_inference keeps adjacent supported f32 activations resident.
// Other layers use their ordinary Sequential forward implementation.
pub fn (mut nn Sequential[f32]) forward_vulkan_inference(input &autograd.Variable[f32]) !&autograd.Variable[f32] {
	validate_vulkan_inference[f32](nn, input)!
	if !layers.vulkan_linear_enabled() {
		return nn.forward(input)
	}

	mut session := VulkanInferenceSession{}
	output := nn.forward_vulkan_inference_resident(input, mut session) or {
		operation_error := err
		session.release() or {
			return error('${operation_error}; additionally failed to release Vulkan session: ${err}')
		}
		return operation_error
	}
	session.release()!
	return output
}

fn (mut nn Sequential[f32]) forward_vulkan_inference_resident(input &autograd.Variable[f32], mut session VulkanInferenceSession) !&autograd.Variable[f32] {
	mut current := input
	for i, layer in nn.info.layers {
		layer_type := if i < nn.info.layer_types.len {
			nn.info.layer_types[i]
		} else {
			''
		}
		if is_vulkan_resident_activation(layer_type) {
			if session.resident == unsafe { nil } {
				session.ensure_device()!
				params := storage.vulkan_params_for_device(session.device)
				session.resident = current.value.vulkan(params)!
			}
			match layer_type {
				'ReLULayer' { session.resident.relu()! }
				'SigmoidLayer' { session.resident.sigmoid()! }
				else { return error('unsupported resident Vulkan activation') }
			}
			continue
		}

		if session.resident != unsafe { nil } {
			host_value := session.resident.cpu()!
			session.release_resident()
			current = input.context.variable(host_value, requires_grad: false)
		}
		current = layer.forward_with_mode(current, nn.training)!
	}

	if session.resident != unsafe { nil } {
		host_value := session.resident.cpu()!
		return input.context.variable(host_value, requires_grad: false)
	}
	return current
}
