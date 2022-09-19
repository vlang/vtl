module autograd

import vtl

pub struct SinGate<T> {
pub:
	a &Variable<T>
}

pub fn sin_gate<T>(a &Variable<T>) &SinGate<T> {
	return &SinGate<T>{
		a: a
	}
}

pub fn (g &SinGate<T>) backward<T>(payload &Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	r0 := gradient.multiply<T>(g.a.value.cos<T>())?
	return [r0]
}

pub fn (g &SinGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) ? {
	a := args[0]

	match a {
		Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			register<T>('Sin', g, result, [a])?
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

pub fn cos_gate<T>(a &Variable<T>) &CosGate<T> {
	return &CosGate<T>{
		a: a
	}
}

pub fn (g &CosGate<T>) backward<T>(payload &Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	r0 := gradient.multiply<T>(g.a.value.sin<T>().multiply_scalar<T>(vtl.cast<T>(-1))?)?
	return [r0]
}

pub fn (g &CosGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) ? {
	a := args[0]

	match a {
		Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			register<T>('Cos', g, result, [a])?
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

pub fn tan_gate<T>(a &Variable<T>) &TanGate<T> {
	return &TanGate<T>{
		a: a
	}
}

pub fn (g &TanGate<T>) backward<T>(payload &Payload<T>) ?[]&vtl.Tensor<T> {
	gradient := payload.variable.grad
	cos := g.a.value.cos<T>()
	r0 := gradient.divide<T>(cos.multiply<T>(cos)?)?
	return [r0]
}

pub fn (g &TanGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) ? {
	a := args[0]

	match a {
		Variable<T> {
			result.grad = vtl.zeros_like<T>(result.value)
			result.requires_grad = true

			register<T>('Tan', g, result, [a])?
		}
		else {
			return error('TanGate: a must be a Variable')
		}
	}
}
