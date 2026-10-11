module autograd

import vtl

// LogGate implements log(x) element-wise.
// backward: grad * (1/x)
pub struct LogGate[T] {
pub:
	a &Variable[T] = unsafe { nil }
}

// log_gate exposes this operation as part of the public API.
pub fn log_gate[T](a &Variable[T]) &LogGate[T] {
	return &LogGate[T]{
		a: a
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &LogGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gradient.divide[T](g.a.value)!
	return [r0]
}

fn log_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&LogGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &LogGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			register[T]('Log', voidptr(g), log_gate_backward_dispatch[T], result, [a])!
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

// abs_gate exposes this operation as part of the public API.
pub fn abs_gate[T](a &Variable[T]) &AbsGate[T] {
	return &AbsGate[T]{
		a: a
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &AbsGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	eps := vtl.cast[T](1e-8)
	abs_a := g.a.value.abs[T]()
	safe_denom := abs_a.map(fn [eps] [T](val T, _ []int) T {
		return if val < eps { eps } else { val }
	})
	sign_a := g.a.value.divide[T](safe_denom)!
	r0 := gradient.multiply[T](sign_a)!
	return [r0]
}

fn abs_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&AbsGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &AbsGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			register[T]('Abs', voidptr(g), abs_gate_backward_dispatch[T], result, [a])!
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

// sqrt_gate exposes this operation as part of the public API.
pub fn sqrt_gate[T](a &Variable[T]) &SqrtGate[T] {
	return &SqrtGate[T]{
		a: a
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &SqrtGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	// d/dx sqrt(x) = 1/(2*sqrt(x)) = gradient * (0.5 / sqrt(x))
	sqrt_a := g.a.value.sqrt[T]()
	two_sqrt := sqrt_a.multiply_scalar[T](vtl.cast[T](2.0))!
	r0 := gradient.divide[T](two_sqrt)!
	return [r0]
}

fn sqrt_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&SqrtGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &SqrtGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			register[T]('Sqrt', voidptr(g), sqrt_gate_backward_dispatch[T], result, [
				a,
			])!
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

// tanh_gate exposes this operation as part of the public API.
pub fn tanh_gate[T](cache &vtl.Tensor[T]) &TanhGate[T] {
	return &TanhGate[T]{
		cache: cache
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &TanhGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	r0 := gradient.nmap([g.cache], fn [T](vals []T, _ []int) T {
		return vals[0] * (vtl.cast[T](1) - vals[1] * vals[1])
	})!
	return [r0]
}

fn tanh_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&TanhGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &TanhGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			register[T]('Tanh', voidptr(g), tanh_gate_backward_dispatch[T], result, [
				a,
			])!
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
	a       &vtl.Tensor[T] = unsafe { nil }
}

// clamp_gate exposes this operation as part of the public API.
pub fn clamp_gate[T](min_val T, max_val T, a &vtl.Tensor[T]) &ClampGate[T] {
	return &ClampGate[T]{
		min_val: min_val
		max_val: max_val
		a:       a
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &ClampGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	input := g.a
	min_val := g.min_val
	max_val := g.max_val
	r0 := gradient.nmap([input], fn [min_val, max_val] [T](vals []T, _ []int) T {
		if vals[1] >= min_val && vals[1] <= max_val {
			return vals[0]
		}
		return vtl.cast[T](0)
	})!
	return [r0]
}

fn clamp_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&ClampGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &ClampGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			register[T]('Clamp', voidptr(g), clamp_gate_backward_dispatch[T], result, [
				a,
			])!
		}
		else {
			return error('ClampGate: a must be a Variable')
		}
	}
}

// NanToNumGate passes gradients only through finite input values, matching
// PyTorch's nan_to_num derivative. The mask is captured during the forward pass
// so later mutation of the source tensor cannot change the backward result.
pub struct NanToNumGate[T] {
pub:
	finite_mask &vtl.Tensor[bool] = unsafe { nil }
}

// nan_to_num_gate exposes the saved finite-value mask to the autograd API.
pub fn nan_to_num_gate[T](finite_mask &vtl.Tensor[bool]) &NanToNumGate[T] {
	return &NanToNumGate[T]{
		finite_mask: finite_mask
	}
}

// backward zeros gradients for values replaced by nan_to_num.
pub fn (g &NanToNumGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	zero_gradient := vtl.zeros_like[T](gradient)
	return [vtl.where[T](g.finite_mask, gradient, zero_gradient)!]
}

fn nan_to_num_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&NanToNumGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache registers the input with the graph for nan_to_num backward.
pub fn (g &NanToNumGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			register[T]('NanToNum', voidptr(g), nan_to_num_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('NanToNumGate: a must be a Variable')
		}
	}
}
