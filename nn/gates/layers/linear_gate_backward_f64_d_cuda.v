module layers

import vtl
import vtl.autograd
import vtl.autograd_cuda

// linear_gate_backward_f64_cuda is only built with `-d cuda` (keeps f32 Gate types clean).
pub fn linear_gate_backward_f64_cuda(gate voidptr, payload voidptr) ![]&vtl.Tensor[f64] {
	g := unsafe { &LinearGate[f64](gate) }
	p := unsafe { &autograd.Payload[f64](payload) }
	grad := p.variable.grad
	mut session := unsafe { &autograd_cuda.DeviceSession(g.input.context.device_session) }
	tensors := autograd_cuda.linear_backward_f64(grad, g.input.value, g.weight.value,
		g.bias.requires_grad, mut session)!
	mut result := [grad, grad, grad]
	if g.input.requires_grad {
		result[0] = tensors[0]
	}
	if g.weight.requires_grad {
		result[1] = tensors[1]
	}
	if g.bias.requires_grad {
		result[2] = tensors[2]
	}
	return result
}
