module layers

import vtl

// linear_gate_backward_f32_try runs Vulkan GEMM backward when VTL_USE_VULKAN=1; else none.
pub fn linear_gate_backward_f32_try(gate voidptr, payload voidptr) ?[]&vtl.Tensor[f32] {
	if !linear_gate_use_vulkan_backward() {
		return none
	}
	return linear_gate_backward_f32_vulkan(gate, payload)!
}
