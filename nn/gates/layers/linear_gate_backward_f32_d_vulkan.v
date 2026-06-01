module layers

import vtl
import vtl.autograd
import vtl.nn.internal

// linear_gate_backward_f32_vulkan exposes this operation as part of the public API.
pub fn linear_gate_backward_f32_vulkan(gate voidptr, payload voidptr) ![]&vtl.Tensor[f32] {
	g := unsafe { &LinearGate[f32](gate) }
	p := unsafe { &autograd.Payload[f32](payload) }
	return internal.linear_backward_vulkan_f32(p.variable.grad, g.input.value, g.weight.value,
		g.bias.requires_grad)
}
