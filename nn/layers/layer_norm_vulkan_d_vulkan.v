module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types
import storage

// LayerNormLayerVulkan is a Vulkan-accelerated layer norm that runs the
// normalisation kernel on GPU (f32 only) and applies gamma/beta on CPU.
pub struct LayerNormLayerVulkan[T] {
	normalized_shape []int
	eps              f32
	params           storage.VulkanStorageParams
pub mut:
	gamma &autograd.Variable[T] = unsafe { nil }
	beta  &autograd.Variable[T] = unsafe { nil }
}

@[params]
pub struct LayerNormVulkanConfig {
	eps    f32  = 1e-5
	affine bool = true
}

pub fn layer_norm_vulkan_layer[T](ctx &autograd.Context[T], normalized_shape []int, params storage.VulkanStorageParams, config LayerNormVulkanConfig) types.Layer[T] {
	mut gamma := unsafe { nil }
	mut beta := unsafe { nil }
	if config.affine {
		gamma = ctx.variable(vtl.ones[T](normalized_shape))
		beta = ctx.variable(vtl.zeros[T](normalized_shape))
	}
	return types.Layer[T](&LayerNormLayerVulkan[T]{
		normalized_shape: normalized_shape
		eps:              config.eps
		params:           params
		gamma:            gamma
		beta:             beta
	})
}

pub fn (layer &LayerNormLayerVulkan[T]) output_shape() []int {
	return layer.normalized_shape
}

pub fn (layer &LayerNormLayerVulkan[T]) variables() []&autograd.Variable[T] {
	if layer.gamma != unsafe { nil } {
		return [layer.gamma, layer.beta]
	}
	return []&autograd.Variable[T]{}
}

pub fn (layer &LayerNormLayerVulkan[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	// GPU normalisation (f32 only; falls back to CPU for other types)
	normed := layernorm_forward_vulkan[T](input.value, layer.eps, layer.params) or {
		// CPU fallback
		internal.layer_norm_forward[T](input.value, layer.gamma.value, layer.beta.value,
			f64(layer.eps))!
	}
	// Apply affine transform on CPU if requested
	mut out := normed
	if layer.gamma != unsafe { nil } {
		out = out.mul(layer.gamma.value)!
		out = out.add(layer.beta.value)!
	}
	mut result := input.context.variable(out)
	if input.requires_grad {
		// Backward still runs on CPU via the existing LayerNormGate
		gate := layernorm_gate[T](input.value, layer.gamma.value, layer.beta.value, f64(layer.eps))
		gate.cache(mut result, input)!
	}
	return result
}
