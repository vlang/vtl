module layers

import vtl
import vtl.nn.internal

// relu_forward_f32 uses Vulkan in-place ReLU when opted in (`-d vulkan`).
pub fn relu_forward_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	if vulkan_linear_enabled() {
		if out := relu_forward_f32_try(x) {
			return out
		}
	}
	return internal.relu[f32](x)
}

// sigmoid_forward_f32 uses Vulkan in-place sigmoid when opted in.
pub fn sigmoid_forward_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	if vulkan_linear_enabled() {
		if out := sigmoid_forward_f32_try(x) {
			return out
		}
	}
	return internal.sigmoid[f32](x)
}
