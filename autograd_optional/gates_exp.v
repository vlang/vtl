module autograd_optional

import vtl

pub struct ExpGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
}

pub fn exp_gate[T](a &Variable[T]) &ExpGate[T] {
	return &ExpGate[T]{
		a: a
	}
}

pub fn (g &ExpGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gradient.multiply[T](g.a.value.exp[T]())!
	return [r0]
}

pub fn (g &ExpGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]

	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true

			register[T]('Exp', g, result, [a])!
		}
		else {
			return error('ExpGate: a must be a Variable')
		}
	}
}
