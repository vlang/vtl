module layers

import vtl

// Stub when not built with `-d vulkan` (keeps f32 Gate backward on CPU).
pub fn linear_gate_backward_f32_try(gate voidptr, payload voidptr) ?[]&vtl.Tensor[f32] {
	_ = gate
	_ = payload
	return none
}
