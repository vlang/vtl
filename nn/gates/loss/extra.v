module loss

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct BCEGate[T] {
pub:
	target       &vtl.Tensor[T] = unsafe { nil }
	from_logits  bool
}

pub fn bce_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T], from_logits bool) &BCEGate[T] {
	return &BCEGate[T]{
		target:      target
		from_logits: from_logits
	}
}

pub fn (g &BCEGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.bce_backward[T](gradient, payload.variable.value, g.target, g.from_logits)!
	return [r0]
}

pub fn (g &BCEGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
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

pub struct HuberLossGate[T] {
pub:
	target &vtl.Tensor[T] = unsafe { nil }
	delta  T
}

pub fn huber_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T], delta T) &HuberLossGate[T] {
	return &HuberLossGate[T]{target: target, delta: delta}
}

pub fn (g &HuberLossGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.huber_backward[T](gradient, payload.variable.value, g.target, g.delta)!
	return [r0]
}

pub fn (g &HuberLossGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
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

pub struct NLLLossGate[T] {
pub:
	target &vtl.Tensor[T] = unsafe { nil }
}

pub fn nll_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) &NLLLossGate[T] {
	return &NLLLossGate[T]{target: target}
}

pub fn (g &NLLLossGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.nll_backward[T](gradient, payload.variable.value, g.target)!
	return [r0]
}

pub fn (g &NLLLossGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
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

pub struct KLDivLossGate[T] {
pub:
	target &vtl.Tensor[T] = unsafe { nil }
}

pub fn kl_div_loss_gate[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) &KLDivLossGate[T] {
	return &KLDivLossGate[T]{target: target}
}

pub fn (g &KLDivLossGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.kl_div_backward[T](gradient, payload.variable.value, g.target)!
	return [r0]
}

pub fn (g &KLDivLossGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
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