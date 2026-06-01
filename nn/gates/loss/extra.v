module loss

import vtl
import vtl.autograd
import vtl.nn.internal

// BCEGate defines a public data structure for this module.
pub struct BCEGate[T] {
pub:
	target      &vtl.Tensor[T] = unsafe { nil }
	from_logits bool
}

// bce_gate exposes this operation as part of the public API.
pub fn bce_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T], from_logits bool) &BCEGate[T] {
	return &BCEGate[T]{
		target:      target
		from_logits: from_logits
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &BCEGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.bce_backward[T](gradient, payload.variable.value, g.target, g.from_logits)!
	return [r0]
}

// cache exposes this operation as part of the public API.
pub fn (g &BCEGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('BCE', g, result, [a])!
		}
		else {
			return error('BCE: cache: invalid argument')
		}
	}
}

// HuberLossGate defines a public data structure for this module.
pub struct HuberLossGate[T] {
pub:
	target &vtl.Tensor[T] = unsafe { nil }
	delta  T
}

// huber_loss_gate exposes this operation as part of the public API.
pub fn huber_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T], delta T) &HuberLossGate[T] {
	return &HuberLossGate[T]{
		target: target
		delta:  delta
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &HuberLossGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.huber_backward[T](gradient, payload.variable.value, g.target, g.delta)!
	return [r0]
}

// cache exposes this operation as part of the public API.
pub fn (g &HuberLossGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Huber', g, result, [a])!
		}
		else {
			return error('Huber: cache: invalid argument')
		}
	}
}

// NLLLossGate defines a public data structure for this module.
pub struct NLLLossGate[T] {
pub:
	target &vtl.Tensor[T] = unsafe { nil }
}

// nll_loss_gate exposes this operation as part of the public API.
pub fn nll_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) &NLLLossGate[T] {
	return &NLLLossGate[T]{
		target: target
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &NLLLossGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.nll_backward[T](gradient, payload.variable.value, g.target)!
	return [r0]
}

// cache exposes this operation as part of the public API.
pub fn (g &NLLLossGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('NLL', g, result, [a])!
		}
		else {
			return error('NLL: cache: invalid argument')
		}
	}
}

// KLDivLossGate defines a public data structure for this module.
pub struct KLDivLossGate[T] {
pub:
	target &vtl.Tensor[T] = unsafe { nil }
}

// kl_div_loss_gate exposes this operation as part of the public API.
pub fn kl_div_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) &KLDivLossGate[T] {
	return &KLDivLossGate[T]{
		target: target
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &KLDivLossGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.kl_div_backward[T](gradient, payload.variable.value, g.target)!
	return [r0]
}

// cache exposes this operation as part of the public API.
pub fn (g &KLDivLossGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('KLDiv', g, result, [a])!
		}
		else {
			return error('KLDiv: cache: invalid argument')
		}
	}
}
