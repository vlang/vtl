module vtl

import math

// all returns whether all array elements evaluate to true.
pub fn (t &Tensor[T]) all[T]() bool {
	mut iter := t.iterator[T]()
	for {
		val, _ := iter.next() or { break }
		bool_value := td[T](val).bool()
		if !bool_value {
			return false
		}
	}
	return true
}

// any returns whether any array elements evaluate to true.
pub fn (t &Tensor[T]) any[T]() bool {
	mut iter := t.iterator[T]()
	for {
		val, _ := iter.next() or { break }
		bool_value := td[T](val).bool()
		if bool_value {
			return true
		}
	}
	return false
}

// is_finite returns true where x is not positive infinity, negative infinity, or NaN;
// false otherwise.
pub fn (t &Tensor[T]) is_finite[T]() &Tensor[bool] {
	mut iter := t.iterator[T]()
	mut ret := empty[bool](t.shape)
	for {
		val, i := iter.next() or { break }
		next_val := math.is_finite(td[T](val).f64())
		ret.set(i, next_val)
	}
	return ret
}

// is_inf reports whether t is an infinity, according to sign.
// If sign > 0, is_inf reports whether t is positive infinity.
// If sign < 0, is_inf reports whether t is negative infinity.
// If sign == 0, is_inf reports whether t is either infinity.
pub fn (t &Tensor[T]) is_inf[T](sign int) &Tensor[bool] {
	mut iter := t.iterator[T]()
	mut ret := empty[bool](t.shape)
	for {
		val, i := iter.next() or { break }
		next_val := math.is_inf(td[T](val).f64(), sign)
		ret.set(i, next_val)
	}
	return ret
}

// is_nan reports whether f is an IEEE 754 ``not-a-number'' value.
pub fn (t &Tensor[T]) is_nan[T]() &Tensor[bool] {
	mut iter := t.iterator[T]()
	mut ret := empty[bool](t.shape)
	for {
		val, i := iter.next() or { break }
		next_val := math.is_nan(td[T](val).f64())
		ret.set(i, next_val)
	}
	return ret
}

// array_equal returns true if input arrays have the same shape and all elements equal.
pub fn (t &Tensor[T]) array_equal[T](other &Tensor[T]) bool {
	if t.shape != other.shape {
		return false
	}
	mut iters, _ := t.iterators[T]([other]) or { return false }
	for {
		vals, _ := iters.next() or { break }
		if vals[0] != vals[1] {
			return false
		}
	}
	return true
}

// array_equiv returns true if input arrays are shape consistent and all elements equal.
// Shape consistent means they are either the same shape,
// or one input array can be broadcasted to create the same shape as the other one.
pub fn (t &Tensor[T]) array_equiv[T](other &Tensor[T]) bool {
	mut iters, _ := t.iterators[T]([other]) or { return false }
	for {
		vals, _ := iters.next() or { break }
		if vals[0] != vals[1] {
			return false
		}
	}
	return true
}

[inline]
fn handle_equal[T](vals []T, _ []int) bool {
	mut equal := true
	for v in vals {
		equal = equal && v == vals[0]
	}
	return equal
}

// equal compares two tensors elementwise
[inline]
pub fn (t &Tensor[T]) equal[T](other &Tensor[T]) !&Tensor[bool] {
	// TODO: Implement using nmap
	mut iters, shape := t.iterators[T]([other])!
	mut ret := empty[bool](shape)
	for {
		vals, i := iters.next() or { break }
		val := handle_equal[T](vals, i)
		ret.set(i, val)
	}
	return ret
}

// not_equal compares two tensors elementwise
[inline]
pub fn (t &Tensor[T]) not_equal[T](other &Tensor[T]) !&Tensor[bool] {
	// TODO: Implement using nmap
	mut iters, shape := t.iterators[T]([other])!
	mut ret := empty[bool](shape)
	for {
		vals, i := iters.next() or { break }
		val := !handle_equal[T](vals, i)
		ret.set(i, val)
	}
	return ret
}

// tolerance compares two tensors elementwise with a given tolerance
[inline]
pub fn (t &Tensor[T]) tolerance[T](other &Tensor[T], tol T) !&Tensor[bool] {
	// TODO: Implement using nmap
	mut iters, shape := t.iterators[T]([other])!
	mut ret := empty[bool](shape)
	for {
		vals, i := iters.next() or { break }
		val := math.tolerance(td[T](vals[0]).f64(), td[T](vals[1]).f64(), td[T](tol).f64())
		ret.set(i, val)
	}
	return ret
}

// close compares two tensors elementwise
[inline]
pub fn (t &Tensor[T]) close[T](other &Tensor[T]) !&Tensor[bool] {
	// TODO: Implement using nmap
	mut iters, shape := t.iterators[T]([other])!
	mut ret := empty[bool](shape)
	for {
		vals, i := iters.next() or { break }
		val := math.close(td[T](vals[0]).f64(), td[T](vals[1]).f64())
		ret.set(i, val)
	}
	return ret
}

// veryclose compares two tensors elementwise
[inline]
pub fn (t &Tensor[T]) veryclose[T](other &Tensor[T]) !&Tensor[bool] {
	// TODO: Implement using nmap
	mut iters, shape := t.iterators[T]([other])!
	mut ret := empty[bool](shape)
	for {
		vals, i := iters.next() or { break }
		val := math.veryclose(td[T](vals[0]).f64(), td[T](vals[1]).f64())
		ret.set(i, val)
	}
	return ret
}

// alike compares two tensors elementwise
[inline]
pub fn (t &Tensor[T]) alike[T](other &Tensor[T]) !&Tensor[bool] {
	// TODO: Implement using nmap
	mut iters, shape := t.iterators[T]([other])!
	mut ret := empty[bool](shape)
	for {
		vals, i := iters.next() or { break }
		val := math.alike(td[T](vals[0]).f64(), td[T](vals[1]).f64())
		ret.set(i, val)
	}
	return ret
}
