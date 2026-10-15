module loss

import vtl
import vtl.autograd
import vtl.nn.internal

// BCEGate defines a public data structure for this module.
pub struct BCEGate[T] {
pub:
	input       &vtl.Tensor[T] = unsafe { nil }
	target      &vtl.Tensor[T] = unsafe { nil }
	from_logits bool
}

// bce_gate exposes this operation as part of the public API.
pub fn bce_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T], from_logits bool) &BCEGate[T] {
	return &BCEGate[T]{
		input:       input
		target:      target
		from_logits: from_logits
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &BCEGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.bce_backward[T](gradient, g.input, g.target, g.from_logits)!
	return [r0]
}

fn bce_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&BCEGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &BCEGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('BCE', voidptr(g), bce_gate_backward_dispatch[T], result, [
				a,
			])!
		}
		else {
			return error('BCE: cache: invalid argument')
		}
	}
}

// HuberLossGate defines a public data structure for this module.
pub struct HuberLossGate[T] {
pub:
	input  &vtl.Tensor[T] = unsafe { nil }
	target &vtl.Tensor[T] = unsafe { nil }
	delta  T
}

// huber_loss_gate exposes this operation as part of the public API.
pub fn huber_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T], delta T) &HuberLossGate[T] {
	return &HuberLossGate[T]{
		input:  input
		target: target
		delta:  delta
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &HuberLossGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.huber_backward[T](gradient, g.input, g.target, g.delta)!
	return [r0]
}

fn huber_loss_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&HuberLossGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &HuberLossGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('Huber', voidptr(g), huber_loss_gate_backward_dispatch[T], result, [
				a,
			])!
		}
		else {
			return error('Huber: cache: invalid argument')
		}
	}
}

// SmoothL1LossGate stores tensors and beta needed for a Smooth L1 backward pass.
pub struct SmoothL1LossGate[T] {
pub:
	input     &vtl.Tensor[T] = unsafe { nil }
	target    &vtl.Tensor[T] = unsafe { nil }
	beta      T
	reduction string = 'mean'
}

// smooth_l1_loss_gate creates a Smooth L1 backward gate.
pub fn smooth_l1_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T], beta T, reduction string) &SmoothL1LossGate[T] {
	return &SmoothL1LossGate[T]{
		input:     input
		target:    target
		beta:      beta
		reduction: reduction
	}
}

// backward computes the Smooth L1 gradient with respect to the prediction.
pub fn (g &SmoothL1LossGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	result := internal.smooth_l1_backward[T](gradient, g.input, g.target, g.beta, g.reduction)!
	return [result]
}

fn smooth_l1_loss_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&SmoothL1LossGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache registers prediction as the differentiable input of Smooth L1.
pub fn (g &SmoothL1LossGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('SmoothL1', voidptr(g), smooth_l1_loss_gate_backward_dispatch[T], result,
				[a])!
		}
		else {
			return error('Smooth L1: cache: invalid argument')
		}
	}
}

// NLLLossGate defines a public data structure for this module.
pub struct NLLLossGate[T] {
pub:
	input        &vtl.Tensor[T] = unsafe { nil }
	target       &vtl.Tensor[T] = unsafe { nil }
	weight       &vtl.Tensor[T] = unsafe { nil }
	ignore_index int            = -1
	reduction    string         = 'mean'
}

// nll_loss_gate exposes this operation as part of the public API.
pub fn nll_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) &NLLLossGate[T] {
	return nll_loss_weighted_gate[T](input, target, unsafe { nil })
}

// nll_loss_weighted_gate creates an NLL gate with per-class weights.
pub fn nll_loss_weighted_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T], weight &vtl.Tensor[T]) &NLLLossGate[T] {
	return nll_loss_weighted_gate_with_options[T](input, target, weight, -1, 'mean')
}

// nll_loss_weighted_gate_with_options configures the weighted NLL backward pass.
pub fn nll_loss_weighted_gate_with_options[T](input &vtl.Tensor[T], target &vtl.Tensor[T], weight &vtl.Tensor[T], ignore_index int, reduction string) &NLLLossGate[T] {
	return &NLLLossGate[T]{
		input:        input
		target:       target
		weight:       weight
		ignore_index: ignore_index
		reduction:    reduction
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &NLLLossGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.nll_backward[T](gradient, g.input, g.target, g.weight, g.ignore_index,
		g.reduction)!
	return [r0]
}

fn nll_loss_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&NLLLossGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &NLLLossGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('NLL', voidptr(g), nll_loss_gate_backward_dispatch[T], result, [
				a,
			])!
		}
		else {
			return error('NLL: cache: invalid argument')
		}
	}
}

// KLDivLossGate defines a public data structure for this module.
pub struct KLDivLossGate[T] {
pub:
	input  &vtl.Tensor[T] = unsafe { nil }
	target &vtl.Tensor[T] = unsafe { nil }
}

// kl_div_loss_gate exposes this operation as part of the public API.
pub fn kl_div_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) &KLDivLossGate[T] {
	return &KLDivLossGate[T]{
		input:  input
		target: target
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &KLDivLossGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.kl_div_backward[T](gradient, g.input, g.target)!
	return [r0]
}

fn kl_div_loss_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&KLDivLossGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &KLDivLossGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			autograd.register[T]('KLDiv', voidptr(g), kl_div_loss_gate_backward_dispatch[T],
				result, [a])!
		}
		else {
			return error('KLDiv: cache: invalid argument')
		}
	}
}
