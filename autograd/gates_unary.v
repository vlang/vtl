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
	// d/dx|x| = sign(x): sign(x) = x / |x|
	// Use: grad * (x / |x|) = grad * sign(x)
	// But we need to handle x=0 where sign is undefined (gradient=0 there)
	abs_a := g.a.value.abs[T]()!
	denom := abs_a.multiply_scalar[T](vtl.cast[T](1))!  // abs_a * 1 for numerical stability
	// sign(x) = x / |x|, but for x=0 we define sign(0)=1
	// So: d/dx|x| = gradient * sign(x)
	// For simplicity, use: gradient * (a / (abs(a) + eps))
	eps := vtl.cast[T](1e-8)
	safe_denom := abs_a.map(fn [eps] [T](val T, i []int) T { return if val < eps { eps } else { val } })!
	sign_a := g.a.value.divide[T](safe_denom)!
	r0 := gradient.multiply[T](sign_a)!
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
	// d/dx sqrt(x) = 1/(2*sqrt(x)) = gradient * (0.5 / sqrt(x))
	sqrt_a := g.a.value.sqrt[T]()!
	half_over_sqrt := sqrt_a.multiply_scalar[T](vtl.cast[T](0.5))!
	r0 := gradient.multiply[T](half_over_sqrt)!
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