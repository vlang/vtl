module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types
import vtl.runtime
import vtl.storage
import vsl.compute as vsl_compute

// LayerNorm normalizes over the last D dimensions of the input.
// E.g. for input [..., D] it computes mean and variance over the last D dims.
@[params]
pub struct LayerNormConfig {
	eps    f64  = 1e-5
	affine bool = true
}

pub struct LayerNormLayer[T] {
	normalized_shape []int
	eps              f64
pub mut:
	gamma &autograd.Variable[T] = unsafe { nil }
	beta  &autograd.Variable[T] = unsafe { nil }
}

// layer_norm_layer creates a LayerNormLayer.
// layer_norm_layer creates a LayerNormLayer.
pub fn layer_norm_layer[T](ctx &autograd.Context[T], normalized_shape []int, config LayerNormConfig) types.Layer[T] {
	mut gamma := unsafe { nil }
	mut beta := unsafe { nil }
	if config.affine {
		gamma = ctx.variable(vtl.ones[T](normalized_shape))
		beta = ctx.variable(vtl.zeros[T](normalized_shape))
	}
	return types.Layer[T](&LayerNormLayer[T]{
		normalized_shape: normalized_shape
		eps:              config.eps
		gamma:            gamma
		beta:             beta
	})
}

pub fn (layer &LayerNormLayer[T]) output_shape() []int {
	return layer.normalized_shape
}

pub fn (layer &LayerNormLayer[T]) variables() []&autograd.Variable[T] {
	if layer.gamma != unsafe { nil } {
		return [layer.gamma, layer.beta]
	}
	return []&autograd.Variable[T]{}
}

pub fn (layer &LayerNormLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	backend := input.context.compute_backend
	strict := input.context.compute_strict
	mut output := internal.layer_norm_forward[T](input.value, layer.gamma.value, layer.beta.value,
		layer.eps)!
	if backend == .vulkan {
		mut dev := runtime.new_vulkan_device()!
		defer {
			dev.release() or {}
		}
		params := storage.new_vulkan_params(dev.device())
		norm := layernorm_forward_vulkan[T](input.value, f32(layer.eps), params)!
		output = norm.multiply[T](layer.gamma.value)!.add[T](layer.beta.value)!
	} else if backend == .auto {
		mut dev := runtime.new_vulkan_device() or { runtime.VulkanDevice{} }
		if !isnil(dev.device()) {
			defer {
				dev.release() or {}
			}
			params := storage.new_vulkan_params(dev.device())
			norm := layernorm_forward_vulkan[T](input.value, f32(layer.eps), params) or {
				unsafe { nil }
			}
			if !isnil(norm) {
				output = norm.multiply[T](layer.gamma.value)!.add[T](layer.beta.value)!
			}
		}
	} else if strict && backend != .cpu {
		available := vsl_compute.available_backends().map(it.str()).join(', ')
		return error('layernorm: backend `${backend}` unavailable for this build. available=[${available}]')
	}
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := layernorm_gate[T](input.value, layer.gamma.value, layer.beta.value, layer.eps)
		gate.cache(mut result, input)!
	}
	return result
}

pub struct LayerNormGate[T] {
	input &vtl.Tensor[T] = unsafe { nil }
	gamma &vtl.Tensor[T] = unsafe { nil }
	beta  &vtl.Tensor[T] = unsafe { nil }
	eps   f64
}

pub fn layernorm_gate[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], eps f64) &LayerNormGate[T] {
	return &LayerNormGate[T]{
		input: input
		gamma: gamma
		beta:  beta
		eps:   eps
	}
}

pub fn (g &LayerNormGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return internal.layer_norm_backward[T](payload.variable.grad, g.input, g.gamma, g.beta, g.eps)
}

pub fn (g &LayerNormGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('LayerNorm', g, result, [a])!
		}
		else {}
	}
}
