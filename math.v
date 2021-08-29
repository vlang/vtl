module vtl

import vtl.storage
import math

fn handle_abs<T>(x T, _ int) T {
	return T(math.abs(f64(x)))
}

// abs returns the elementwise abs of an tensor
[inline]
pub fn abs<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_abs)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_abs<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_acos<T>(x T, _ int) T {
	return T(math.acos(f64(x)))
}

// acos returns the elementwise acos of an tensor
[inline]
pub fn acos<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_acos)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_acos<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_acosh<T>(x T, _ int) T {
	return T(math.acosh(f64(x)))
}

// acosh returns the elementwise acosh of an tensor
[inline]
pub fn acosh<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_acosh)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_acosh<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_asin<T>(x T, _ int) T {
	return T(math.asin(f64(x)))
}

// asin returns the elementwise asin of an tensor
[inline]
pub fn asin<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_asin)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_asin<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_asinh<T>(x T, _ int) T {
	return T(math.asinh(f64(x)))
}

// asinh returns the elementwise asinh of an tensor
[inline]
pub fn asinh<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_asinh)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_asinh<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_atan<T>(x T, _ int) T {
	return T(math.atan(f64(x)))
}

// atan returns the elementwise atan of an tensor
[inline]
pub fn atan<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_atan)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_atan<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_atan2<T>(xs []T, _ int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.atan2(x, y))
}

// atan2 returns the atan2 elementwise of two tensors
[inline]
pub fn atan2<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_atan2, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_atan2<T>(vals, pos)
		storage.storage_set<T>(ret.data, pos, val)
	}
	return ret
}

fn handle_atanh<T>(x T, _ int) T {
	return T(math.atanh(f64(x)))
}

// atanh returns the elementwise atanh of an tensor
[inline]
pub fn atanh<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_atanh)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_atanh<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_cbrt<T>(x T, _ int) T {
	return T(math.cbrt(f64(x)))
}

// cbrt returns the elementwise cbrt of an tensor
[inline]
pub fn cbrt<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_cbrt)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_cbrt<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_ceil<T>(x T, _ int) T {
	return T(math.ceil(f64(x)))
}

// ceil returns the elementwise ceil of an tensor
[inline]
pub fn ceil<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_ceil)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_ceil<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_cos<T>(x T, _ int) T {
	return T(math.cos(f64(x)))
}

// cos returns the elementwise cos of an tensor
[inline]
pub fn cos<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_cos)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_cos<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_cosh<T>(x T, _ int) T {
	return T(math.cosh(f64(x)))
}

// cosh returns the elementwise cosh of an tensor
[inline]
pub fn cosh<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_cosh)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_cosh<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_cot<T>(x T, _ int) T {
	return T(math.cot(f64(x)))
}

// cot returns the elementwise cot of an tensor
[inline]
pub fn cot<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_cot)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_cot<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_degrees<T>(x T, _ int) T {
	return T(math.degrees(f64(x)))
}

// degrees returns the elementwise degrees of an tensor
[inline]
pub fn degrees<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_degrees)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_degrees<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_erf<T>(x T, _ int) T {
	return T(math.erf(f64(x)))
}

// erf returns the elementwise erf of an tensor
[inline]
pub fn erf<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_erf)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_erf<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_erfc<T>(x T, _ int) T {
	return T(math.erfc(f64(x)))
}

// erfc returns the elementwise erfc of an tensor
[inline]
pub fn erfc<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_erfc)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_erfc<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_exp<T>(x T, _ int) T {
	return T(math.exp(f64(x)))
}

// exp returns the elementwise exp of an tensor
[inline]
pub fn exp<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_exp)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_exp<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_exp2<T>(x T, _ int) T {
	return T(math.exp2(f64(x)))
}

// exp2 returns the elementwise exp2 of an tensor
[inline]
pub fn exp2<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_exp2)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_exp2<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_expm1<T>(x T, _ int) T {
	return T(math.expm1(f64(x)))
}

// expm1 returns the elementwise expm1 of an tensor
[inline]
pub fn expm1<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_expm1)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_expm1<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_f32_bits<T>(x T, _ int) T {
	return T(math.f32_bits(f32(x)))
}

// f32_bits returns the elementwise f32_bits of an tensor
[inline]
pub fn f32_bits<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_f32_bits)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_f32_bits<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_f32_from_bits<T>(x T, _ int) T {
	return T(math.f32_from_bits(u32(x)))
}

// f32_from_bits returns the elementwise f32_from_bits of an tensor
[inline]
pub fn f32_from_bits<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_f32_from_bits)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_f32_from_bits<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_f64_bits<T>(x T, _ int) T {
	return T(math.f64_bits(f64(x)))
}

// f64_bits returns the elementwise f64_bits of an tensor
[inline]
pub fn f64_bits<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_f64_bits)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_f64_bits<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_f64_from_bits<T>(x T, _ int) T {
	return T(math.f64_from_bits(u64(x)))
}

// f64_from_bits returns the elementwise f64_from_bits of an tensor
[inline]
pub fn f64_from_bits<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_f64_from_bits)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_f64_from_bits<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_factorial<T>(x T, _ int) T {
	return T(math.factorial(f64(x)))
}

// factorial returns the elementwise factorial of an tensor
[inline]
pub fn factorial<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_factorial)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_factorial<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_floor<T>(x T, _ int) T {
	return T(math.floor(f64(x)))
}

// floor returns the elementwise floor of an tensor
[inline]
pub fn floor<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_floor)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_floor<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_fmod<T>(xs []T, _ int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.fmod(x, y))
}

// fmod returns the fmod elementwise of two tensors
[inline]
pub fn fmod<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_fmod, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_fmod<T>(vals, pos)
		storage.storage_set<T>(ret.data, pos, val)
	}
	return ret
}

fn handle_gamma<T>(x T, _ int) T {
	return T(math.gamma(f64(x)))
}

// gamma returns the elementwise gamma of an tensor
[inline]
pub fn gamma<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_gamma)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_gamma<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_gcd<T>(xs []T, _ int) T {
	x := i64(xs[0])
	y := i64(xs[1])
	return T(math.gcd(x, y))
}

// gcd returns the gcd elementwise of two tensors
[inline]
pub fn gcd<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_gcd, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_gcd<T>(vals, pos)
		storage.storage_set<T>(ret.data, pos, val)
	}
	return ret
}

fn handle_hypot<T>(xs []T, _ int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.hypot(x, y))
}

// hypot returns the hypot elementwise of two tensors
[inline]
pub fn hypot<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_hypot, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_hypot<T>(vals, pos)
		storage.storage_set<T>(ret.data, pos, val)
	}
	return ret
}

fn handle_lcm<T>(xs []T, _ int) T {
	x := i64(xs[0])
	y := i64(xs[1])
	return T(math.lcm(x, y))
}

// lcm returns the lcm elementwise of two tensors
[inline]
pub fn lcm<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_lcm, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_lcm<T>(vals, pos)
		storage.storage_set<T>(ret.data, pos, val)
	}
	return ret
}

fn handle_log<T>(x T, _ int) T {
	return T(math.log(f64(x)))
}

// log returns the elementwise log of an tensor
[inline]
pub fn log<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_log)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_log<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_log10<T>(x T, _ int) T {
	return T(math.log10(f64(x)))
}

// log10 returns the elementwise log10 of an tensor
[inline]
pub fn log10<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_log10)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_log10<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_log1p<T>(x T, _ int) T {
	return T(math.log1p(f64(x)))
}

// log1p returns the elementwise log1p of an tensor
[inline]
pub fn log1p<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_log1p)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_log1p<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_log2<T>(x T, _ int) T {
	return T(math.log2(f64(x)))
}

// log2 returns the elementwise log2 of an tensor
[inline]
pub fn log2<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_log2)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_log2<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_log_factorial<T>(x T, _ int) T {
	return T(math.log_factorial(f64(x)))
}

// log_factorial returns the elementwise log_factorial of an tensor
[inline]
pub fn log_factorial<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_log_factorial)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_log_factorial<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_log_gamma<T>(x T, _ int) T {
	return T(math.log_gamma(f64(x)))
}

// log_gamma returns the elementwise log_gamma of an tensor
[inline]
pub fn log_gamma<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_log_gamma)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_log_gamma<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_log_n<T>(xs []T, _ int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.log_n(x, y))
}

// log_n returns the log_n elementwise of two tensors
[inline]
pub fn log_n<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_log_n, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_log_n<T>(vals, pos)
		storage.storage_set<T>(ret.data, pos, val)
	}
	return ret
}

fn handle_max<T>(xs []T, _ int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.max(x, y))
}

// max returns the max elementwise of two tensors
[inline]
pub fn max<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_max, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_max<T>(vals, pos)
		storage.storage_set<T>(ret.data, pos, val)
	}
	return ret
}

fn handle_min<T>(xs []T, _ int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.min(x, y))
}

// min returns the min elementwise of two tensors
[inline]
pub fn min<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_min, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_min<T>(vals, pos)
		storage.storage_set<T>(ret.data, pos, val)
	}
	return ret
}

fn handle_nextafter<T>(xs []T, _ int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.nextafter(x, y))
}

// nextafter returns the nextafter elementwise of two tensors
[inline]
pub fn nextafter<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_nextafter, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_nextafter<T>(vals, pos)
		storage.storage_set<T>(ret.data, pos, val)
	}
	return ret
}

fn handle_nextafterf32<T>(xs []T, _ int) T {
	x := f32(xs[0])
	y := f32(xs[1])
	return T(math.nextafterf32(x, y))
}

// nextafter32 returns the nextafter32 elementwise of two tensors
[inline]
pub fn nextafter32<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_nextafterf32, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_nextafterf32<T>(valpos, i)
		storage.storage_set<T>(ret.data, pos, val)
	}
	return ret
}

fn handle_pow<T>(xs []T, _ int) T {
	x := f64(xs[0])
	y := f64(xs[1])
	return T(math.pow(x, y))
}

// pow returns the pow elementwise of two tensors
[inline]
pub fn pow<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_pow, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_pow<T>(vals, pos)
		storage.storage_set<T>(ret.data, pos, val)
	}
	return ret
}

fn handle_pow10<T>(x T, _ int) T {
	return T(math.pow10(f64(x)))
}

// pow10 returns the elementwise pow10 of an tensor
[inline]
pub fn pow10<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_pow10)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_pow10<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_radians<T>(x T, _ int) T {
	return T(math.radians(f64(x)))
}

// radians returns the elementwise deg2rad of an tensor
[inline]
pub fn radians<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_radians)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_radians<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_round<T>(x T, _ int) T {
	return T(math.round(f64(x)))
}

// round rounds elements of an tensor elementwise
[inline]
pub fn round<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_round)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_round<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_round_to_even<T>(x T, _ int) T {
	return T(math.round_to_even(f64(x)))
}

// round_to_even round_to_evens elements of an tensor elementwise
[inline]
pub fn round_to_even<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_round_to_even)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_round_to_even<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_sin<T>(x T, _ int) T {
	return T(math.sin(f64(x)))
}

// sin returns the elementwise sin of an tensor
[inline]
pub fn sin<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_sin)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_sin<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_sinh<T>(x T, _ int) T {
	return T(math.sinh(f64(x)))
}

// sinh returns the elementwise sinh of an tensor
[inline]
pub fn sinh<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_sinh)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_sinh<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_sqrt<T>(x T, _ int) T {
	return T(math.sqrt(f64(x)))
}

// sqrt returns the elementwise square root of an tensor
[inline]
pub fn sqrt<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_sqrt)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_sqrt<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_tan<T>(x T, _ int) T {
	return T(math.tan(f64(x)))
}

// tan returns the elementwise tan of an tensor
[inline]
pub fn tan<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_tan)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_tan<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_tanh<T>(x T, _ int) T {
	return T(math.tanh(f64(x)))
}

// tanh returns the elementwise tanh of an tensor
[inline]
pub fn tanh<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_tanh)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_tanh<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}

fn handle_trunc<T>(x T, _ int) T {
	return T(math.trunc(f64(x)))
}

// trunc returns the elementwise trunc of an tensor
[inline]
pub fn trunc<T>(t &Tensor<T>) &Tensor<T> {
	// @todo: Implement using t.map
	// return t.map<T>(handle_trunc)
	mut ret := new_tensor_like<T>(t)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_trunc<T>(val, pos)
		storage.storage_set<T>(ret.data, pos, next_val)
	}
	return ret
}
