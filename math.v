module vtl

import math

// abs returns the elementwise abs of an tensor

// abs exposes this operation as part of the public API.

// abs exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) abs[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		// TODO: Figure out a way to do this without casting to f64
		return cast[T](math.abs(td(x).f64()))
	})
}

// acos returns the elementwise acos of an tensor

// acos exposes this operation as part of the public API.

// acos exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) acos[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.acos(td(x).f64()))
	})
}

// acosh returns the elementwise acosh of an tensor

// acosh exposes this operation as part of the public API.

// acosh exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) acosh[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.acosh(td(x).f64()))
	})
}

// asin returns the elementwise asin of an tensor

// asin exposes this operation as part of the public API.

// asin exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) asin[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.asin(td(x).f64()))
	})
}

// asinh returns the elementwise asinh of an tensor

// asinh exposes this operation as part of the public API.

// asinh exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) asinh[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.asinh(td(x).f64()))
	})
}

// atan returns the elementwise atan of an tensor

// atan exposes this operation as part of the public API.

// atan exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) atan[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.atan(td(x).f64()))
	})
}

// atan2 returns the atan2 elementwise of two tensors

// atan2 exposes this operation as part of the public API.

// atan2 exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) atan2[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap([b], fn [T](xs []T, _ []int) T {
		x := xs[0]
		y := xs[1]
		return cast[T](math.atan2(td(x).f64(), td(y).f64()))
	})
}

// atanh returns the elementwise atanh of an tensor

// atanh exposes this operation as part of the public API.

// atanh exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) atanh[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.atanh(td(x).f64()))
	})
}

// cbrt returns the elementwise cbrt of an tensor

// cbrt exposes this operation as part of the public API.

// cbrt exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) cbrt[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.cbrt(td(x).f64()))
	})
}

// ceil returns the elementwise ceil of an tensor

// ceil exposes this operation as part of the public API.

// ceil exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) ceil[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.ceil(td(x).f64()))
	})
}

// cos returns the elementwise cos of an tensor

// cos exposes this operation as part of the public API.

// cos exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) cos[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.cos(td(x).f64()))
	})
}

// cosh returns the elementwise cosh of an tensor

// cosh exposes this operation as part of the public API.

// cosh exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) cosh[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.cosh(td(x).f64()))
	})
}

// cot returns the elementwise cot of an tensor

// cot exposes this operation as part of the public API.

// cot exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) cot[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.cot(td(x).f64()))
	})
}

// degrees returns the elementwise degrees of an tensor

// degrees exposes this operation as part of the public API.

// degrees exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) degrees[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.degrees(td(x).f64()))
	})
}

// erf returns the elementwise erf of an tensor

// erf exposes this operation as part of the public API.

// erf exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) erf[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.erf(td(x).f64()))
	})
}

// erfc returns the elementwise erfc of an tensor

// erfc exposes this operation as part of the public API.

// erfc exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) erfc[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.erfc(td(x).f64()))
	})
}

// exp returns the elementwise exp of an tensor

// exp exposes this operation as part of the public API.

// exp exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) exp[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.exp(td(x).f64()))
	})
}

// exp2 returns the elementwise exp2 of an tensor

// exp2 exposes this operation as part of the public API.

// exp2 exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) exp2[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.exp2(td(x).f64()))
	})
}

// expm1 returns the elementwise expm1 of an tensor

// expm1 exposes this operation as part of the public API.

// expm1 exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) expm1[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.expm1(td(x).f64()))
	})
}

// f32_bits returns the elementwise f32_bits of an tensor

// f32_bits exposes this operation as part of the public API.

// f32_bits exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) f32_bits[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.f32_bits(td(x).f32()))
	})
}

// f32_from_bits returns the elementwise f32_from_bits of an tensor

// f32_from_bits exposes this operation as part of the public API.

// f32_from_bits exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) f32_from_bits[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.f32_from_bits(td(x).u32()))
	})
}

// f64_bits returns the elementwise f64_bits of an tensor

// f64_bits exposes this operation as part of the public API.

// f64_bits exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) f64_bits[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.f64_bits(td(x).f64()))
	})
}

// f64_from_bits returns the elementwise f64_from_bits of an tensor

// f64_from_bits exposes this operation as part of the public API.

// f64_from_bits exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) f64_from_bits[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.f64_from_bits(td(x).u64()))
	})
}

// factorial returns the elementwise factorial of an tensor

// factorial exposes this operation as part of the public API.

// factorial exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) factorial[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.factorial(td(x).f64()))
	})
}

// floor returns the elementwise floor of an tensor

// floor exposes this operation as part of the public API.

// floor exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) floor[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.floor(td(x).f64()))
	})
}

// fmod returns the fmod elementwise of two tensors

// fmod exposes this operation as part of the public API.

// fmod exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) fmod[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn [T](xs []T, _ []int) T {
		x := xs[0]
		y := xs[1]
		return cast[T](math.fmod(td(x).f64(), td(y).f64()))
	})
}

// gamma returns the elementwise gamma of an tensor

// gamma exposes this operation as part of the public API.

// gamma exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) gamma[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.gamma(td(x).f64()))
	})
}

// gcd returns the gcd elementwise of two tensors

// gcd exposes this operation as part of the public API.

// gcd exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) gcd[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn [T](xs []T, _ []int) T {
		x := xs[0]
		y := xs[1]
		return cast[T](math.gcd(td(x).i64(), td(y).i64()))
	})
}

// hypot returns the hypot elementwise of two tensors

// hypot exposes this operation as part of the public API.

// hypot exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) hypot[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn [T](xs []T, _ []int) T {
		x := xs[0]
		y := xs[1]
		return cast[T](math.hypot(td(x).f64(), td(y).f64()))
	})
}

// lcm returns the lcm elementwise of two tensors

// lcm exposes this operation as part of the public API.

// lcm exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) lcm[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn [T](xs []T, _ []int) T {
		x := xs[0]
		y := xs[1]
		return cast[T](math.lcm(td(x).i64(), td(y).i64()))
	})
}

// log returns the elementwise log of an tensor

// log exposes this operation as part of the public API.

// log exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) log[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.log(td(x).f64()))
	})
}

// log10 returns the elementwise log10 of an tensor

// log10 exposes this operation as part of the public API.

// log10 exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) log10[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.log10(td(x).f64()))
	})
}

// log1p returns the elementwise log1p of an tensor

// log1p exposes this operation as part of the public API.

// log1p exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) log1p[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.log1p(td(x).f64()))
	})
}

// log2 returns the elementwise log2 of an tensor

// log2 exposes this operation as part of the public API.

// log2 exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) log2[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.log2(td(x).f64()))
	})
}

// log_factorial returns the elementwise log_factorial of an tensor

// log_factorial exposes this operation as part of the public API.

// log_factorial exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) log_factorial[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.log_factorial(td(x).f64()))
	})
}

// log_gamma returns the elementwise log_gamma of an tensor

// log_gamma exposes this operation as part of the public API.

// log_gamma exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) log_gamma[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.log_gamma(td(x).f64()))
	})
}

// log_n returns the log_n elementwise of two tensors

// log_n exposes this operation as part of the public API.

// log_n exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) log_n[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn [T](xs []T, _ []int) T {
		x := xs[0]
		y := xs[1]
		return cast[T](math.log_n(td(x).f64(), td(y).f64()))
	})
}

// max returns the max elementwise of two tensors

// max exposes this operation as part of the public API.

// max exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) max[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn [T](xs []T, _ []int) T {
		return math.max(xs[0], xs[1])
	})
}

// min returns the min elementwise of two tensors

// min exposes this operation as part of the public API.

// min exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) min[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn [T](xs []T, _ []int) T {
		return math.min(xs[0], xs[1])
	})
}

// nextafter returns the nextafter elementwise of two tensors

// nextafter exposes this operation as part of the public API.

// nextafter exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) nextafter[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn [T](xs []T, _ []int) T {
		x := xs[0]
		y := xs[1]
		return cast[T](math.nextafter(td(x).f64(), td(y).f64()))
	})
}

// nextafter32 returns the nextafter32 elementwise of two tensors

// nextafter32 exposes this operation as part of the public API.

// nextafter32 exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) nextafter32[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn [T](xs []T, _ []int) T {
		x := xs[0]
		y := xs[1]
		return cast[T](math.nextafter32(td(x).f32(), td(y).f32()))
	})
}

// pow returns the pow elementwise of two tensors

// pow exposes this operation as part of the public API.

// pow exposes this operation as part of the public API.
@[inline]
pub fn (a &Tensor[T]) pow[T](b &Tensor[T]) !&Tensor[T] {
	return a.nmap[T]([b], fn [T](xs []T, _ []int) T {
		x := xs[0]
		y := xs[1]
		return cast[T](math.pow(td(x).f64(), td(y).f64()))
	})
}

// pow10 returns the elementwise pow10 of an tensor

// pow10 exposes this operation as part of the public API.

// pow10 exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) pow10[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.pow10(td(x).int()))
	})
}

// radians returns the elementwise deg2rad of an tensor

// radians exposes this operation as part of the public API.

// radians exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) radians[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.radians(td(x).f64()))
	})
}

// round rounds elements of an tensor elementwise

// round exposes this operation as part of the public API.

// round exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) round[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.round(td(x).f64()))
	})
}

// round_to_even round_to_evens elements of an tensor elementwise

// round_to_even exposes this operation as part of the public API.

// round_to_even exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) round_to_even[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.round_to_even(td(x).f64()))
	})
}

// sin returns the elementwise sin of an tensor

// sin exposes this operation as part of the public API.

// sin exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) sin[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.sin(td(x).f64()))
	})
}

// sinh returns the elementwise sinh of an tensor

// sinh exposes this operation as part of the public API.

// sinh exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) sinh[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.sinh(td(x).f64()))
	})
}

// sqrt returns the elementwise square root of an tensor

// sqrt exposes this operation as part of the public API.

// sqrt exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) sqrt[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.sqrt(td(x).f64()))
	})
}

// tan returns the elementwise tan of an tensor

// tan exposes this operation as part of the public API.

// tan exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) tan[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.tan(td(x).f64()))
	})
}

// tanh returns the elementwise tanh of an tensor

// tanh exposes this operation as part of the public API.

// tanh exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) tanh[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.tanh(td(x).f64()))
	})
}

// trunc returns the elementwise trunc of an tensor

// trunc exposes this operation as part of the public API.

// trunc exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) trunc[T]() &Tensor[T] {
	return t.map(fn [T](x T, _ []int) T {
		return cast[T](math.trunc(td(x).f64()))
	})
}
