module autograd

import vtl

// LogGate implements log(x) element-wise.
// backward: grad * (1/x)
pub struct LogGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
}

pub fn log_gate[T](a &Variable[T]) &LogGate[T] {
	return &LogGate[T]{a: a}
}

pub fn (g &LogGate[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gradient.divide[T](g.a.value)!
	return [r0]
}

pub fn (g &LogGate[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Log', g, result, [a])!
		}
		else {
			return error('LogGate: a must be a Variable')
		}
	}
}

// AbsGate implements abs(x) element-wise.
// backward: grad * sign(x)
pub struct AbsGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
}

pub fn abs_gate[T](a &Variable[T]) &AbsGate[T] {
	return &AbsGate[T]{a: a}
}

pub fn (g &AbsGate[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	// sign(x) = 1 if x > 0, -1 if x < 0, 0 if x == 0
	r0 := gradient.multiply[T](g.a.value.abs[T]().divide_scalar[T](vtl.cast[T](g.a.value.get_nth(0)))!
	return [r0]
}

pub fn (g &AbsGate[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Abs', g, result, [a])!
		}
		else {
			return error('AbsGate: a must be a Variable')
		}
	}
}

// SqrtGate implements sqrt(x) element-wise.
// backward: grad * (1 / (2 * sqrt(x)))
pub struct SqrtGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
}

pub fn sqrt_gate[T](a &Variable[T]) &SqrtGate[T] {
	return &SqrtGate[T]{a: a}
}

pub fn (g &SqrtGate[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	// d/dx sqrt(x) = 1/(2*sqrt(x)) = 0.5 / sqrt(x)
	sqrt_a := g.a.value.sqrt[T]()
	r0 := gradient.multiply[T](sqrt_a.multiply_scalar[T](vtl.cast[T](0.5))!
	return [r0]
}

pub fn (g &SqrtGate[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Sqrt', g, result, [a])!
		}
		else {
			return error('SqrtGate: a must be a Variable')
		}
	}
}

// TanhGate implements tanh(x) element-wise.
// backward: grad * (1 - tanh(x)^2) = grad * (1 - cached^2)
pub struct TanhGate[T] {
pub:
	cache &vtl.Tensor[T] = unsafe { nil }
}

pub fn tanh_gate[T](cache &vtl.Tensor[T]) &TanhGate[T] {
	return &TanhGate[T]{cache: cache}
}

pub fn (g &TanhGate[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gradient.nmap([g.cache], fn [T](vals []T, _ []int) T {
		return vals[0] * (vtl.cast[T](1) - vals[1] * vals[1])
	})!
	return [r0]
}

pub fn (g &TanhGate[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Tanh', g, result, [a])!
		}
		else {
			return error('TanhGate: a must be a Variable')
		}
	}
}

// ClampGate implements clamp(x, min, max) element-wise.
// backward: grad where x is within [min, max], 0 otherwise
pub struct ClampGate[T] {
pub:
	min_val T
	max_val T
}

pub fn clamp_gate[T](min_val T, max_val T) &ClampGate[T] {
	return &ClampGate[T]{min_val: min_val, max_val: max_val}
}

pub fn (g &ClampGate[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	cached := payload.variable.value
	r0 := gradient.nmap([cached], fn [T](vals []T, _ []int) T {
		// Pass gradient where value is within bounds, else 0
		if vals[1] >= g.min_val && vals[1] <= g.max_val {
			return vals[0]
		}
		return vtl.cast[T](0)
	})!
	return [r0]
}

pub fn (g &ClampGate[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('Clamp', g, result, [a])!
		}
		else {
			return error('ClampGate: a must be a Variable')
		}
	}
}