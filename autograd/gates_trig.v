module autograd

import vtl

pub struct SinGate<T> {
pub:
	a &Variable<T>
}

pub fn new_sin_gate<T>(a &Variable<T>) &SinGate<T> {
	return &SinGate<T>{
		a: a
	}
}

pub fn (g &SinGate<T>) backward<T>(payload &Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	r0 := vtl.multiply<T>(gradient, vtl.cos<T>(g.a.value))
	return [r0]
}

pub fn (g &SinGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) ? {
	a := args[0]

	match a {
		Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			register<T>('Sin', g, result, a)?
		}
		else {
			return error('SinGate: a must be a Variable')
		}
	}
}

pub struct CosGate<T> {
pub:
	a &Variable<T>
}

pub fn new_cos_gate<T>(a &Variable<T>) &CosGate<T> {
	return &CosGate<T>{
		a: a
	}
}

pub fn (g &CosGate<T>) backward<T>(payload &Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	r0 := vtl.multiply<T>(gradient, vtl.multiply_scalar<T>(vtl.sin<T>(g.a.value), T(-1)))
	return [r0]
}

pub fn (g &CosGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) ? {
	a := args[0]

	match a {
		Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			register<T>('Cos', g, result, a)?
		}
		else {
			return error('CosGate: a must be a Variable')
		}
	}
}

pub struct TanGate<T> {
pub:
	a &Variable<T>
}

pub fn new_tan_gate<T>(a &Variable<T>) &TanGate<T> {
	return &TanGate<T>{
		a: a
	}
}

pub fn (g &TanGate<T>) backward<T>(payload &Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	cos := vtl.cos<T>(g.a.value)
	r0 := vtl.divide<T>(gradient, vtl.multiply<T>(cos, cos))
	return [r0]
}

pub fn (g &TanGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) ? {
	a := args[0]

	match a {
		Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			register<T>('Tan', g, result, a)?
		}
		else {
			return error('TanGate: a must be a Variable')
		}
	}
}
