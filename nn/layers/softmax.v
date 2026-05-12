module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.types
import vtl.nn.gates.activation as activation_gates
import vsl.compute as vsl_compute
import vsl.vulkan
import vtl.storage

// SoftmaxLayer applies softmax activation over the last dimension of the input.
// input shape: [..., n_classes]  →  output shape: [..., n_classes]
// Softmax output sums to 1 along the last dimension.
pub struct SoftmaxLayer[T] {
	dim int
}

@[params]
pub struct SoftmaxLayerConfig {
	dim int = -1 // dimension to apply softmax over; -1 means last dimension
}

pub fn softmax_layer[T](ctx &autograd.Context[T], config SoftmaxLayerConfig) types.Layer[T] {
	return types.Layer[T](&SoftmaxLayer[T]{
		dim: config.dim
	})
}

pub fn (layer &SoftmaxLayer[T]) output_shape() []int {
	return []int{}
}

pub fn (layer &SoftmaxLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &SoftmaxLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	// Compute softmax along dim (-1 means last dim)
	dim := if layer.dim == -1 { input.value.shape.len - 1 } else { layer.dim }
	backend := input.context.compute_backend
	strict := input.context.compute_strict
	mut output := internal.softmax_forward[T](input.value, dim)!
	if input.value.shape.len == 1 && dim == 0 {
		if backend == .vulkan {
			mut dev := vulkan.new_device()!
			defer {
				dev.release() or {}
			}
			output = softmax_forward_vulkan[T](input.value, storage.new_vulkan_params(dev))!
		} else if backend == .auto {
			mut dev := vulkan.new_device() or { unsafe { nil } }
			if !isnil(dev) {
				defer {
					dev.release() or {}
				}
				output = softmax_forward_vulkan[T](input.value, storage.new_vulkan_params(dev)) or {
					output
				}
			}
		} else if strict && backend != .cpu {
			available := vsl_compute.available_backends().map(it.str()).join(', ')
			return error('softmax: backend `${backend}` unavailable for this build. available=[${available}]')
		}
	} else if strict && backend != .cpu && backend != .auto {
		available := vsl_compute.available_backends().map(it.str()).join(', ')
		return error('softmax: backend `${backend}` unsupported for this shape/dim. available=[${available}]')
	}
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation_gates.softmax_gate[T](input.value, dim)
		gate.cache(mut result, input)!
	}
	return result
}
