module vtl

import math

fn handle_abs<T>(x T, _ []int) T {
	return T(math.abs(f64(x)))
}

// abs returns the elementwise abs of an tensor
[inline]
pub fn (t &Tensor<T>) abs<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_abs<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_acos<T>(x T, _ []int) T {
	return T(math.acos(f64(x)))
}

// acos returns the elementwise acos of an tensor
[inline]
pub fn (t &Tensor<T>) acos<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_acos<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_acosh<T>(x T, _ []int) T {
	return T(math.acosh(f64(x)))
}

// acosh returns the elementwise acosh of an tensor
[inline]
pub fn (t &Tensor<T>) acosh<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_acosh<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_asin<T>(x T, _ []int) T {
	return T(math.asin(f64(x)))
}

// asin returns the elementwise asin of an tensor
[inline]
pub fn (t &Tensor<T>) asin<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_asin<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_asinh<T>(x T, _ []int) T {
	return T(math.asinh(f64(x)))
}

// asinh returns the elementwise asinh of an tensor
[inline]
pub fn (t &Tensor<T>) asinh<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_asinh<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_atan<T>(x T, _ []int) T {
	return T(math.atan(f64(x)))
}

// atan returns the elementwise atan of an tensor
[inline]
pub fn (t &Tensor<T>) atan<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_atan<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_atan2<T>(xs []T, _ []int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.atan2(x, y))
}

// atan2 returns the atan2 elementwise of two tensors
[inline]
pub fn (a &Tensor<T>) atan2<T>(b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_atan2, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := handle_atan2<T>(vals, i)
		ret.set(i, val)
	}
	return ret
}

fn handle_atanh<T>(x T, _ []int) T {
	return T(math.atanh(f64(x)))
}

// atanh returns the elementwise atanh of an tensor
[inline]
pub fn (t &Tensor<T>) atanh<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_atanh<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_cbrt<T>(x T, _ []int) T {
	return T(math.cbrt(f64(x)))
}

// cbrt returns the elementwise cbrt of an tensor
[inline]
pub fn (t &Tensor<T>) cbrt<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_cbrt<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_ceil<T>(x T, _ []int) T {
	return T(math.ceil(f64(x)))
}

// ceil returns the elementwise ceil of an tensor
[inline]
pub fn (t &Tensor<T>) ceil<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_ceil<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_cos<T>(x T, _ []int) T {
	return T(math.cos(f64(x)))
}

// cos returns the elementwise cos of an tensor
[inline]
pub fn (t &Tensor<T>) cos<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_cos<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_cosh<T>(x T, _ []int) T {
	return T(math.cosh(f64(x)))
}

// cosh returns the elementwise cosh of an tensor
[inline]
pub fn (t &Tensor<T>) cosh<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_cosh<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_cot<T>(x T, _ []int) T {
	return T(math.cot(f64(x)))
}

// cot returns the elementwise cot of an tensor
[inline]
pub fn (t &Tensor<T>) cot<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_cot<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_degrees<T>(x T, _ []int) T {
	return T(math.degrees(f64(x)))
}

// degrees returns the elementwise degrees of an tensor
[inline]
pub fn (t &Tensor<T>) degrees<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_degrees<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_erf<T>(x T, _ []int) T {
	return T(math.erf(f64(x)))
}

// erf returns the elementwise erf of an tensor
[inline]
pub fn (t &Tensor<T>) erf<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_erf<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_erfc<T>(x T, _ []int) T {
	return T(math.erfc(f64(x)))
}

// erfc returns the elementwise erfc of an tensor
[inline]
pub fn (t &Tensor<T>) erfc<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_erfc<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_exp<T>(x T, _ []int) T {
	return T(math.exp(f64(x)))
}

// exp returns the elementwise exp of an tensor
[inline]
pub fn (t &Tensor<T>) exp<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_exp<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_exp2<T>(x T, _ []int) T {
	return T(math.exp2(f64(x)))
}

// exp2 returns the elementwise exp2 of an tensor
[inline]
pub fn (t &Tensor<T>) exp2<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_exp2<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_expm1<T>(x T, _ []int) T {
	return T(math.expm1(f64(x)))
}

// expm1 returns the elementwise expm1 of an tensor
[inline]
pub fn (t &Tensor<T>) expm1<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_expm1<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_f32_bits<T>(x T, _ []int) T {
	return T(math.f32_bits(f32(x)))
}

// f32_bits returns the elementwise f32_bits of an tensor
[inline]
pub fn (t &Tensor<T>) f32_bits<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_f32_bits<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_f32_from_bits<T>(x T, _ []int) T {
	return T(math.f32_from_bits(u32(x)))
}

// f32_from_bits returns the elementwise f32_from_bits of an tensor
[inline]
pub fn (t &Tensor<T>) f32_from_bits<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_f32_from_bits<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_f64_bits<T>(x T, _ []int) T {
	return T(math.f64_bits(f64(x)))
}

// f64_bits returns the elementwise f64_bits of an tensor
[inline]
pub fn (t &Tensor<T>) f64_bits<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_f64_bits<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_f64_from_bits<T>(x T, _ []int) T {
	return T(math.f64_from_bits(u64(x)))
}

// f64_from_bits returns the elementwise f64_from_bits of an tensor
[inline]
pub fn (t &Tensor<T>) f64_from_bits<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_f64_from_bits<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_factorial<T>(x T, _ []int) T {
	return T(math.factorial(f64(x)))
}

// factorial returns the elementwise factorial of an tensor
[inline]
pub fn (t &Tensor<T>) factorial<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_factorial<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_floor<T>(x T, _ []int) T {
	return T(math.floor(f64(x)))
}

// floor returns the elementwise floor of an tensor
[inline]
pub fn (t &Tensor<T>) floor<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_floor<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_fmod<T>(xs []T, _ []int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.fmod(x, y))
}

// fmod returns the fmod elementwise of two tensors
[inline]
pub fn (a &Tensor<T>) fmod<T>(b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_fmod, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := handle_fmod<T>(vals, i)
		ret.set(i, val)
	}
	return ret
}

fn handle_gamma<T>(x T, _ []int) T {
	return T(math.gamma(f64(x)))
}

// gamma returns the elementwise gamma of an tensor
[inline]
pub fn (t &Tensor<T>) gamma<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_gamma<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_gcd<T>(xs []T, _ []int) T {
	x := i64(xs[0])
	y := i64(xs[1])
	return T(math.gcd(x, y))
}

// gcd returns the gcd elementwise of two tensors
[inline]
pub fn (a &Tensor<T>) gcd<T>(b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_gcd, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := handle_gcd<T>(vals, i)
		ret.set(i, val)
	}
	return ret
}

fn handle_hypot<T>(xs []T, _ []int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.hypot(x, y))
}

// hypot returns the hypot elementwise of two tensors
[inline]
pub fn (a &Tensor<T>) hypot<T>(b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_hypot, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := handle_hypot<T>(vals, i)
		ret.set(i, val)
	}
	return ret
}

fn handle_lcm<T>(xs []T, _ []int) T {
	x := i64(xs[0])
	y := i64(xs[1])
	return T(math.lcm(x, y))
}

// lcm returns the lcm elementwise of two tensors
[inline]
pub fn (a &Tensor<T>) lcm<T>(b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_lcm, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := handle_lcm<T>(vals, i)
		ret.set(i, val)
	}
	return ret
}

fn handle_log<T>(x T, _ []int) T {
	return T(math.log(f64(x)))
}

// log returns the elementwise log of an tensor
[inline]
pub fn (t &Tensor<T>) log<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_log<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_log10<T>(x T, _ []int) T {
	return T(math.log10(f64(x)))
}

// log10 returns the elementwise log10 of an tensor
[inline]
pub fn (t &Tensor<T>) log10<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_log10<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_log1p<T>(x T, _ []int) T {
	return T(math.log1p(f64(x)))
}

// log1p returns the elementwise log1p of an tensor
[inline]
pub fn (t &Tensor<T>) log1p<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_log1p<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_log2<T>(x T, _ []int) T {
	return T(math.log2(f64(x)))
}

// log2 returns the elementwise log2 of an tensor
[inline]
pub fn (t &Tensor<T>) log2<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_log2<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_log_factorial<T>(x T, _ []int) T {
	return T(math.log_factorial(f64(x)))
}

// log_factorial returns the elementwise log_factorial of an tensor
[inline]
pub fn (t &Tensor<T>) log_factorial<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_log_factorial<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_log_gamma<T>(x T, _ []int) T {
	return T(math.log_gamma(f64(x)))
}

// log_gamma returns the elementwise log_gamma of an tensor
[inline]
pub fn (t &Tensor<T>) log_gamma<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_log_gamma<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_log_n<T>(xs []T, _ []int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.log_n(x, y))
}

// log_n returns the log_n elementwise of two tensors
[inline]
pub fn (a &Tensor<T>) log_n<T>(b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_log_n, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := handle_log_n<T>(vals, i)
		ret.set(i, val)
	}
	return ret
}

fn handle_max<T>(xs []T, _ []int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.max(x, y))
}

// max returns the max elementwise of two tensors
[inline]
pub fn (a &Tensor<T>) max<T>(b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_max, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := handle_max<T>(vals, i)
		ret.set(i, val)
	}
	return ret
}

fn handle_min<T>(xs []T, _ []int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.min(x, y))
}

// min returns the min elementwise of two tensors
[inline]
pub fn (a &Tensor<T>) min<T>(b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_min, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := handle_min<T>(vals, i)
		ret.set(i, val)
	}
	return ret
}

fn handle_nextafter<T>(xs []T, _ []int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.nextafter(x, y))
}

// nextafter returns the nextafter elementwise of two tensors
[inline]
pub fn (a &Tensor<T>) nextafter<T>(b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_nextafter, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := handle_nextafter<T>(vals, i)
		ret.set(i, val)
	}
	return ret
}

fn handle_nextafter32<T>(xs []T, _ []int) T {
	x := f32(xs[0])
	y := f32(xs[1])
	return T(math.nextafter32(x, y))
}

// nextafter32 returns the nextafter32 elementwise of two tensors
[inline]
pub fn (a &Tensor<T>) nextafter32<T>(b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_nextafter32, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := handle_nextafter32<T>(vals, i)
		ret.set(i, val)
	}
	return ret
}

fn handle_pow<T>(xs []T, _ []int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.pow(x, y))
}

// pow returns the pow elementwise of two tensors
[inline]
pub fn (a &Tensor<T>) pow<T>(b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_pow, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := handle_pow<T>(vals, i)
		ret.set(i, val)
	}
	return ret
}

fn handle_pow10<T>(x T, _ []int) T {
	return T(math.pow10(int(x)))
}

// pow10 returns the elementwise pow10 of an tensor
[inline]
pub fn (t &Tensor<T>) pow10<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_pow10<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_radians<T>(x T, _ []int) T {
	return T(math.radians(f64(x)))
}

// radians returns the elementwise deg2rad of an tensor
[inline]
pub fn (t &Tensor<T>) radians<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_radians<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_round<T>(x T, _ []int) T {
	return T(math.round(f64(x)))
}

// round rounds elements of an tensor elementwise
[inline]
pub fn (t &Tensor<T>) round<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_round<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_round_to_even<T>(x T, _ []int) T {
	return T(math.round_to_even(f64(x)))
}

// round_to_even round_to_evens elements of an tensor elementwise
[inline]
pub fn (t &Tensor<T>) round_to_even<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_round_to_even<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_sin<T>(x T, _ []int) T {
	return T(math.sin(f64(x)))
}

// sin returns the elementwise sin of an tensor
[inline]
pub fn (t &Tensor<T>) sin<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_sin<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_sinh<T>(x T, _ []int) T {
	return T(math.sinh(f64(x)))
}

// sinh returns the elementwise sinh of an tensor
[inline]
pub fn (t &Tensor<T>) sinh<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_sinh<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_sqrt<T>(x T, _ []int) T {
	return T(math.sqrt(f64(x)))
}

// sqrt returns the elementwise square root of an tensor
[inline]
pub fn (t &Tensor<T>) sqrt<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_sqrt<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_tan<T>(x T, _ []int) T {
	return T(math.tan(f64(x)))
}

// tan returns the elementwise tan of an tensor
[inline]
pub fn (t &Tensor<T>) tan<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_tan<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_tanh<T>(x T, _ []int) T {
	return T(math.tanh(f64(x)))
}

// tanh returns the elementwise tanh of an tensor
[inline]
pub fn (t &Tensor<T>) tanh<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_tanh<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}

fn handle_trunc<T>(x T, _ []int) T {
	return T(math.trunc(f64(x)))
}

// trunc returns the elementwise trunc of an tensor
[inline]
pub fn (t &Tensor<T>) trunc<T>() &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_trunc<T>(val, i)
		ret.set(i, next_val)
	}
	return ret
}
