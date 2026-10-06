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

// softplus_forward_f32 uses Vulkan when opted in, then falls back to CPU.
pub fn softplus_forward_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	if vulkan_linear_enabled() {
		if out := softplus_forward_f32_try(x) {
			return out
		}
	}
	return internal.softplus[f32](x)
}

// selu_forward_f32 uses Vulkan when opted in, then falls back to CPU.
pub fn selu_forward_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	if vulkan_linear_enabled() {
		if out := selu_forward_f32_try(x) {
			return out
		}
	}
	return internal.selu[f32](x)
}

// hardswish_forward_f32 uses Vulkan when opted in, then falls back to CPU.
pub fn hardswish_forward_f32(x &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	if vulkan_linear_enabled() {
		if out := hardswish_forward_f32_try(x) {
			return out
		}
	}
	return internal.hardswish[f32](x)
}
