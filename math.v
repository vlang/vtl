module vtl

import vsl.vmath as math

pub type F64Fn = fn (x f64) f64

// wraps float function with Num param
// fn math_call(f F64Fn) MapFn {
// return fn (x Num, _ int) Num {
// val := num_as_type<f64>(x)
// ret := f(val)
// return Num(ret)
// }
// }
// abs returns the elementwise abs of an tensor
[inline]
pub fn abs(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.abs(val)
		return Num(ret)
	})
}

// acos returns the elementwise acos of an tensor
[inline]
pub fn acos(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.acos(val)
		return Num(ret)
	})
}

// asin returns the elementwise asin of an tensor
[inline]
pub fn asin(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.asin(val)
		return Num(ret)
	})
}

// atan returns the elementwise atan of an tensor
[inline]
pub fn atan(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.atan(val)
		return Num(ret)
	})
}

// cbrt returns the elementwise cbrt of an tensor
[inline]
pub fn cbrt(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.cbrt(val)
		return Num(ret)
	})
}

// ceil returns the elementwise ceil of an tensor
[inline]
pub fn ceil(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.ceil(val)
		return Num(ret)
	})
}

// cos returns the elementwise cos of an tensor
[inline]
pub fn cos(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.cos(val)
		return Num(ret)
	})
}

// exp returns the elementwise exp of an tensor
[inline]
pub fn exp(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.exp(val)
		return Num(ret)
	})
}

// erf returns the elementwise erf of an tensor
[inline]
pub fn erf(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.erf(val)
		return Num(ret)
	})
}

// exp2 returns the elementwise exp2 of an tensor
[inline]
pub fn exp2(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.exp2(val)
		return Num(ret)
	})
}

// floor returns the elementwise floor of an tensor
[inline]
pub fn floor(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.floor(val)
		return Num(ret)
	})
}

// gamma returns the elementwise gamma of an tensor
[inline]
pub fn gamma(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.gamma(val)
		return Num(ret)
	})
}

// log10 returns the elementwise log10 of an tensor
[inline]
pub fn log10(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.log10(val)
		return Num(ret)
	})
}

// log_gamma returns the elementwise log_gamma of an tensor
[inline]
pub fn log_gamma(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.log_gamma(val)
		return Num(ret)
	})
}

// radians returns the elementwise deg2rad of an tensor
[inline]
pub fn radians(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.radians(val)
		return Num(ret)
	})
}

// round rounds elements of an tensor elementwise
[inline]
pub fn round(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.round(val)
		return Num(ret)
	})
}

// sin returns the elementwise sin of an tensor
[inline]
pub fn sin(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.sin(val)
		return Num(ret)
	})
}

// sinh returns the elementwise sinh of an tensor
[inline]
pub fn sinh(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.sinh(val)
		return Num(ret)
	})
}

// sqrt returns the elementwise square root of an tensor
[inline]
pub fn sqrt(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.sqrt(val)
		return Num(ret)
	})
}

// tan returns the elementwise tan of an tensor
[inline]
pub fn tan(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.tan(val)
		return Num(ret)
	})
}

// tanh returns the elementwise tanh of an tensor
[inline]
pub fn tanh(t Tensor) Tensor {
	return t.map(fn (x Num, _ int) Num {
		val := num_as_type<f64>(x)
		ret := math.tanh(val)
		return Num(ret)
	})
}
