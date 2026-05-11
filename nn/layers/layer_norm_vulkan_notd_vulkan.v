module layers

// Stub for non-Vulkan builds — LayerNormLayerVulkan falls through to CPU.
import vtl
import vtl.autograd
import vtl.nn.types
import storage

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
	_ = params
	return layer_norm_layer[T](ctx, normalized_shape, LayerNormConfig{
		eps:    f64(config.eps)
		affine: config.affine
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
	return error('LayerNormLayerVulkan.forward: Vulkan not enabled (compile with -d vulkan)')
}
