module vtl

import vsl.vmath as math
import vtl.etype

// abs returns the elementwise abs of an tensor
[inline]
pub fn abs(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.abs(val)
		return etype.Num(ret)
	})
}

// acos returns the elementwise acos of an tensor
[inline]
pub fn acos(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.acos(val)
		return etype.Num(ret)
	})
}

// asin returns the elementwise asin of an tensor
[inline]
pub fn asin(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.asin(val)
		return etype.Num(ret)
	})
}

// atan returns the elementwise atan of an tensor
[inline]
pub fn atan(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.atan(val)
		return etype.Num(ret)
	})
}

// atan2 returns the atan2 elementwise of two tensors
[inline]
pub fn atan2(a Tensor, b Tensor) Tensor {
	f := fn (xs []etype.Num, _ int) etype.Num {
		x := etype.num_as_type<f64>(xs[0])
		y := etype.num_as_type<f64>(xs[1])
		ret := math.atan2(x, y)
		return etype.Num(ret)
	}
	return a.nmap(f, b)
}

// cbrt returns the elementwise cbrt of an tensor
[inline]
pub fn cbrt(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.cbrt(val)
		return etype.Num(ret)
	})
}

// ceil returns the elementwise ceil of an tensor
[inline]
pub fn ceil(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.ceil(val)
		return etype.Num(ret)
	})
}

// cos returns the elementwise cos of an tensor
[inline]
pub fn cos(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.cos(val)
		return etype.Num(ret)
	})
}

// exp returns the elementwise exp of an tensor
[inline]
pub fn exp(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.exp(val)
		return etype.Num(ret)
	})
}

// erf returns the elementwise erf of an tensor
[inline]
pub fn erf(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.erf(val)
		return etype.Num(ret)
	})
}

// exp2 returns the elementwise exp2 of an tensor
[inline]
pub fn exp2(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.exp2(val)
		return etype.Num(ret)
	})
}

// floor returns the elementwise floor of an tensor
[inline]
pub fn floor(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.floor(val)
		return etype.Num(ret)
	})
}

// fmod returns the fmod elementwise of two tensors
[inline]
pub fn fmod(a Tensor, b Tensor) Tensor {
	f := fn (xs []etype.Num, _ int) etype.Num {
		x := etype.num_as_type<f64>(xs[0])
		y := etype.num_as_type<f64>(xs[1])
		ret := math.fmod(x, y)
		return etype.Num(ret)
	}
	return a.nmap(f, b)
}

// gamma returns the elementwise gamma of an tensor
[inline]
pub fn gamma(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.gamma(val)
		return etype.Num(ret)
	})
}

// hypot returns the hypot elementwise of two tensors
[inline]
pub fn hypot(a Tensor, b Tensor) Tensor {
	f := fn (xs []etype.Num, _ int) etype.Num {
		x := etype.num_as_type<f64>(xs[0])
		y := etype.num_as_type<f64>(xs[1])
		ret := math.hypot(x, y)
		return etype.Num(ret)
	}
	return a.nmap(f, b)
}

// log10 returns the elementwise log10 of an tensor
[inline]
pub fn log10(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.log10(val)
		return etype.Num(ret)
	})
}

// log_gamma returns the elementwise log_gamma of an tensor
[inline]
pub fn log_gamma(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.log_gamma(val)
		return etype.Num(ret)
	})
}

// log_n returns the log_n elementwise of two tensors
[inline]
pub fn log_n(a Tensor, b Tensor) Tensor {
	f := fn (xs []etype.Num, _ int) etype.Num {
		x := etype.num_as_type<f64>(xs[0])
		y := etype.num_as_type<f64>(xs[1])
		ret := math.log_n(x, y)
		return etype.Num(ret)
	}
	return a.nmap(f, b)
}

// max returns the max elementwise of two tensors
[inline]
pub fn max(a Tensor, b Tensor) Tensor {
	f := fn (xs []etype.Num, _ int) etype.Num {
		x := etype.num_as_type<f64>(xs[0])
		y := etype.num_as_type<f64>(xs[1])
		ret := math.max(x, y)
		return etype.Num(ret)
	}
	return a.nmap(f, b)
}

// min returns the min elementwise of two tensors
[inline]
pub fn min(a Tensor, b Tensor) Tensor {
	f := fn (xs []etype.Num, _ int) etype.Num {
		x := etype.num_as_type<f64>(xs[0])
		y := etype.num_as_type<f64>(xs[1])
		ret := math.min(x, y)
		return etype.Num(ret)
	}
	return a.nmap(f, b)
}

// pow returns the pow elementwise of two tensors
[inline]
pub fn pow(a Tensor, b Tensor) Tensor {
	f := fn (xs []etype.Num, _ int) etype.Num {
		x := etype.num_as_type<f64>(xs[0])
		y := etype.num_as_type<f64>(xs[1])
		ret := math.pow(x, y)
		return etype.Num(ret)
	}
	return a.nmap(f, b)
}

// radians returns the elementwise deg2rad of an tensor
[inline]
pub fn radians(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.radians(val)
		return etype.Num(ret)
	})
}

// round rounds elements of an tensor elementwise
[inline]
pub fn round(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.round(val)
		return etype.Num(ret)
	})
}

// sin returns the elementwise sin of an tensor
[inline]
pub fn sin(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.sin(val)
		return etype.Num(ret)
	})
}

// sinh returns the elementwise sinh of an tensor
[inline]
pub fn sinh(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.sinh(val)
		return etype.Num(ret)
	})
}

// sqrt returns the elementwise square root of an tensor
[inline]
pub fn sqrt(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.sqrt(val)
		return etype.Num(ret)
	})
}

// tan returns the elementwise tan of an tensor
[inline]
pub fn tan(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.tan(val)
		return etype.Num(ret)
	})
}

// tanh returns the elementwise tanh of an tensor
[inline]
pub fn tanh(t Tensor) Tensor {
	return t.map(fn (x etype.Num, _ int) etype.Num {
		val := etype.num_as_type<f64>(x)
		ret := math.tanh(val)
		return etype.Num(ret)
	})
}
