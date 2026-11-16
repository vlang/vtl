module models

import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.storage
import vsl.vulkan

// forward_vulkan_inference keeps adjacent supported f32 activations resident.
// Other layers use their ordinary Sequential forward implementation.
pub fn (mut nn Sequential[f32]) forward_vulkan_inference(input &autograd.Variable[f32]) !&autograd.Variable[f32] {
	validate_vulkan_inference[f32](nn, input)!
	if !layers.vulkan_linear_enabled() {
		return nn.forward(input)
	}

	mut device := &vulkan.Device(unsafe { nil })
	mut resident := &vtl.VulkanTensor[f32](unsafe { nil })
	defer {
		if resident != unsafe { nil } {
			resident.release()
		}
		if device != unsafe { nil } {
			device.release() or { panic(err) }
		}
	}
	mut current := input
	for i, layer in nn.info.layers {
		layer_type := if i < nn.info.layer_types.len {
			nn.info.layer_types[i]
		} else {
			''
		}
		if is_vulkan_resident_activation(layer_type) {
			if resident == unsafe { nil } {
				if device == unsafe { nil } {
					device = vulkan.new_device()!
				}
				params := storage.vulkan_params_for_device(device)
				resident = current.value.vulkan(params)!
			}
			match layer_type {
				'ReLULayer' { resident.relu()! }
				'SigmoidLayer' { resident.sigmoid()! }
				else { return error('unsupported resident Vulkan activation') }
			}
			continue
		}

		if resident != unsafe { nil } {
			host_value := resident.cpu()!
			resident.release()
			resident = unsafe { nil }
			current = input.context.variable(host_value, requires_grad: false)
		}
		current = layer.forward_with_mode(current, nn.training)!
	}

	if resident != unsafe { nil } {
		host_value := resident.cpu()!
		resident.release()
		resident = unsafe { nil }
		return input.context.variable(host_value, requires_grad: false)
	}
	return current
}
