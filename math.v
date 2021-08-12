module vtl

import vsl.vmath as math

// abs returns the elementwise abs of an tensor
[inline]
pub fn abs<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.abs(f64(x)))
	})
}

// acos returns the elementwise acos of an tensor
[inline]
pub fn acos<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.acos(f64(x)))
	})
}

// acosh returns the elementwise acosh of an tensor
[inline]
pub fn acosh<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.acosh(f64(x)))
	})
}

// asin returns the elementwise asin of an tensor
[inline]
pub fn asin<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.asin(f64(x)))
	})
}

// asinh returns the elementwise asinh of an tensor
[inline]
pub fn asinh<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.asinh(f64(x)))
	})
}

// atan returns the elementwise atan of an tensor
[inline]
pub fn atan<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.atan(f64(x)))
	})
}

// atan2 returns the atan2 elementwise of two tensors
[inline]
pub fn atan2<T, U>(a &Tensor<T>, b &Tensor<T>) &Tensor {
	f := fn (xs []T, _ int) U {
		x := f64(xs[0])
		y := f64(xs[1])
		return U(math.atan2(x, y))
	}
	return a.nmap<T, U>(f, b)
}

// atanh returns the elementwise atanh of an tensor
[inline]
pub fn atanh<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.atanh(f64(x)))
	})
}

// cbrt returns the elementwise cbrt of an tensor
[inline]
pub fn cbrt<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.cbrt(f64(x)))
	})
}

// ceil returns the elementwise ceil of an tensor
[inline]
pub fn ceil<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.ceil(f64(x)))
	})
}

// cos returns the elementwise cos of an tensor
[inline]
pub fn cos<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.cos(f64(x)))
	})
}

// cosh returns the elementwise cosh of an tensor
[inline]
pub fn cosh<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.cosh(f64(x)))
	})
}

// cot returns the elementwise cot of an tensor
[inline]
pub fn cot<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.cot(f64(x)))
	})
}

// degrees returns the elementwise degrees of an tensor
[inline]
pub fn degrees<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.degrees(f64(x)))
	})
}

// erf returns the elementwise erf of an tensor
[inline]
pub fn erf<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.erf(f64(x)))
	})
}

// erfc returns the elementwise erfc of an tensor
[inline]
pub fn erfc<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.erfc(f64(x)))
	})
}

// exp returns the elementwise exp of an tensor
[inline]
pub fn exp<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.exp(f64(x)))
	})
}

// exp2 returns the elementwise exp2 of an tensor
[inline]
pub fn exp2<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.exp2(f64(x)))
	})
}

// expm1 returns the elementwise expm1 of an tensor
[inline]
pub fn expm1<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.expm1(f64(x)))
	})
}

// f32_bits returns the elementwise f32_bits of an tensor
[inline]
pub fn f32_bits<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.f32_bits(f32(x)))
	})
}

// f32_from_bits returns the elementwise f32_from_bits of an tensor
[inline]
pub fn f32_from_bits<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.f32_from_bits(u32(x)))
	})
}

// f64_bits returns the elementwise f64_bits of an tensor
[inline]
pub fn f64_bits<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.f64_bits(f64(x)))
	})
}

// f64_from_bits returns the elementwise f64_from_bits of an tensor
[inline]
pub fn f64_from_bits<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.f64_from_bits(u64(x)))
	})
}

// factorial returns the elementwise factorial of an tensor
[inline]
pub fn factorial<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.factorial(f64(x)))
	})
}

// floor returns the elementwise floor of an tensor
[inline]
pub fn floor<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.floor(f64(x)))
	})
}

// fmod returns the fmod elementwise of two tensors
[inline]
pub fn fmod<T, U>(a &Tensor<T>, b &Tensor<T>) &Tensor {
	f := fn (xs []T, _ int) U {
		x := f64(xs[0])
		y := f64(xs[1])
		return U(math.fmod(x, y))
	}
	return a.nmap<T, U>(f, b)
}

// gamma returns the elementwise gamma of an tensor
[inline]
pub fn gamma<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.gamma(f64(x)))
	})
}

// gcd returns the gcd elementwise of two tensors
[inline]
pub fn gcd<T, U>(a &Tensor<T>, b &Tensor<T>) &Tensor {
	f := fn (xs []T, _ int) U {
		x := i64(xs[0])
		y := i64(xs[1])
		return U(math.gcd(x, y))
	}
	return a.nmap<T, U>(f, b)
}

// hypot returns the hypot elementwise of two tensors
[inline]
pub fn hypot<T, U>(a &Tensor<T>, b &Tensor<T>) &Tensor {
	f := fn (xs []T, _ int) U {
		x := f64(xs[0])
		y := f64(xs[1])
		return U(math.hypot(x, y))
	}
	return a.nmap<T, U>(f, b)
}

// lcm returns the lcm elementwise of two tensors
[inline]
pub fn lcm<T, U>(a &Tensor<T>, b &Tensor<T>) &Tensor {
	f := fn (xs []T, _ int) U {
		x := i64(xs[0])
		y := i64(xs[1])
		return U(math.lcm(x, y))
	}
	return a.nmap<T, U>(f, b)
}

// log returns the elementwise log of an tensor
[inline]
pub fn log<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.log(f64(x)))
	})
}

// log10 returns the elementwise log10 of an tensor
[inline]
pub fn log10<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.log10(f64(x)))
	})
}

// log1p returns the elementwise log1p of an tensor
[inline]
pub fn log1p<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.log1p(f64(x)))
	})
}

// log2 returns the elementwise log2 of an tensor
[inline]
pub fn log2<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.log2(f64(x)))
	})
}

// log_factorial returns the elementwise log_factorial of an tensor
[inline]
pub fn log_factorial<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.log_factorial(f64(x)))
	})
}

// log_gamma returns the elementwise log_gamma of an tensor
[inline]
pub fn log_gamma<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.log_gamma(f64(x)))
	})
}

// log_n returns the log_n elementwise of two tensors
[inline]
pub fn log_n<T, U>(a &Tensor<T>, b &Tensor<T>) &Tensor {
	f := fn (xs []T, _ int) U {
		x := f64(xs[0])
		y := f64(xs[1])
		return U(math.log_n(x, y))
	}
	return a.nmap<T, U>(f, b)
}

// max returns the max elementwise of two tensors
[inline]
pub fn max<T, U>(a &Tensor<T>, b &Tensor<T>) &Tensor {
	f := fn (xs []T, _ int) U {
		x := f64(xs[0])
		y := f64(xs[1])
		return U(math.max(x, y))
	}
	return a.nmap<T, U>(f, b)
}

// min returns the min elementwise of two tensors
[inline]
pub fn min<T, U>(a &Tensor<T>, b &Tensor<T>) &Tensor {
	f := fn (xs []T, _ int) U {
		x := f64(xs[0])
		y := f64(xs[1])
		return U(math.min(x, y))
	}
	return a.nmap<T, U>(f, b)
}

// nextafter returns the nextafter elementwise of two tensors
[inline]
pub fn nextafter<T, U>(a &Tensor<T>, b &Tensor<T>) &Tensor {
	f := fn (xs []T, _ int) U {
		x := f64(xs[0])
		y := f64(xs[1])
		return U(math.nextafter(x, y))
	}
	return a.nmap<T, U>(f, b)
}

// nextafter32 returns the nextafter32 elementwise of two tensors
[inline]
pub fn nextafter32<T, U>(a &Tensor<T>, b &Tensor<T>) &Tensor {
	f := fn (xs []T, _ int) U {
		x := f32(xs[0])
		y := f32(xs[1])
		return U(math.nextafterf32(x, y))
	}
	return a.nmap<T, U>(f, b)
}

// pow returns the pow elementwise of two tensors
[inline]
pub fn pow<T, U>(a &Tensor<T>, b &Tensor<T>) &Tensor {
	f := fn (xs []T, _ int) U {
		x := f64(xs[0])
		y := f64(xs[1])
		return U(math.pow(x, y))
	}
	return a.nmap<T, U>(f, b)
}

// pow10 returns the elementwise pow10 of an tensor
[inline]
pub fn pow10<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.pow10(f64(x)))
	})
}

// radians returns the elementwise deg2rad of an tensor
[inline]
pub fn radians<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.radians(f64(x)))
	})
}

// round rounds elements of an tensor elementwise
[inline]
pub fn round<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.round(f64(x)))
	})
}

// round_to_even round_to_evens elements of an tensor elementwise
[inline]
pub fn round_to_even<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.round_to_even(f64(x)))
	})
}

fn handle_sin<T, U>(x T, _ int) U {
	return U(math.sin(f64(x)))
}

// sin returns the elementwise sin of an tensor
[inline]
pub fn sin<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(handle_sin)
}

// sinh returns the elementwise sinh of an tensor
[inline]
pub fn sinh<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.sinh(f64(x)))
	})
}

// sqrt returns the elementwise square root of an tensor
[inline]
pub fn sqrt<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.sqrt(f64(x)))
	})
}

// tan returns the elementwise tan of an tensor
[inline]
pub fn tan<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.tan(f64(x)))
	})
}

// tanh returns the elementwise tanh of an tensor
[inline]
pub fn tanh<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.tanh(f64(x)))
	})
}

// trunc returns the elementwise trunc of an tensor
[inline]
pub fn trunc<T, U>(t &Tensor<T>) &Tensor<U> {
	return t.map<T, U>(fn (x T, _ int) U {
		return U(math.trunc(f64(x)))
	})
}
