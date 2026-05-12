module layers

import vtl

// attention_forward_vulkan stub: not available without Vulkan.
pub fn attention_forward_vulkan[T](q &vtl.Tensor[T], k &vtl.Tensor[T], v &vtl.Tensor[T], head_dim int) !&vtl.Tensor[T] {
	return error('attention_forward_vulkan: not available without -d vulkan flag')
}

pub fn attention_scores_vulkan[T](q &vtl.Tensor[T], k &vtl.Tensor[T], head_dim int) !&vtl.Tensor[T] {
	return error('attention_scores_vulkan: not available without -d vulkan flag')
}

pub fn attention_apply_values_vulkan[T](weights &vtl.Tensor[T], v &vtl.Tensor[T], head_dim int) !&vtl.Tensor[T] {
	return error('attention_apply_values_vulkan: not available without -d vulkan flag')
}
