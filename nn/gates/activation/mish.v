module activation

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct MishGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

pub fn mish_gate[T](cache &vtl.Tensor[T]) &MishGate[T] {
	return &MishGate[T]{cache: cache}
}

pub fn (g &MishGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := internal.deriv_mish[T](gradient, g.cache)!
	return [r0]
}

pub fn (g &MishGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('Mish', g, result, [a])!
		}
		else {
			return error('Mish: cache: invalid argument')
		}
	}
}