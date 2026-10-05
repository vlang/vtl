module loss

import vtl
import vtl.autograd
import vtl.nn.internal

// AdditionalLossGate routes gradients for MAE, hinge, and focal losses.
pub struct AdditionalLossGate[T] {
pub:
	input       &vtl.Tensor[T] = unsafe { nil }
	target      &vtl.Tensor[T] = unsafe { nil }
	kind        string
	alpha       f64
	gamma       f64
	from_logits bool
}

// additional_loss_gate creates the internal gate used by additional loss functions.
pub fn additional_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T], kind string, alpha f64, gamma f64, from_logits bool) &AdditionalLossGate[T] {
	return &AdditionalLossGate[T]{ input: input, target: target, kind: kind, alpha: alpha, gamma: gamma, from_logits: from_logits }
}

pub fn (g &AdditionalLossGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	value := g.input
	match g.kind {
		'l1' { return [internal.l1_backward[T](gradient, value, g.target)!] }
		'hinge' { return [internal.hinge_backward[T](gradient, value, g.target)!] }
		'focal' {
			return [internal.focal_backward[T](gradient, value, g.target, g.alpha, g.gamma, g.from_logits)!]
		}
		else { return error('unknown additional loss: ${g.kind}') }
	}
}

fn additional_loss_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&AdditionalLossGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &AdditionalLossGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('AdditionalLoss:${g.kind}', voidptr(g), additional_loss_gate_backward_dispatch[T], result, [a])!
		}
		else { return error('AdditionalLoss: cache: invalid argument') }
	}
}
