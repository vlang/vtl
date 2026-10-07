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

// logical_and evaluates element truth values with standard tensor broadcasting.
pub fn (a &Tensor[T]) logical_and[T](b &Tensor[T]) !&Tensor[bool] {
	return logical_binary[T](a, b, fn [T](left T, right T) bool {
		return td[T](left).bool() && td[T](right).bool()
	})
}

// logical_or evaluates element truth values with standard tensor broadcasting.
pub fn (a &Tensor[T]) logical_or[T](b &Tensor[T]) !&Tensor[bool] {
	return logical_binary[T](a, b, fn [T](left T, right T) bool {
		return td[T](left).bool() || td[T](right).bool()
	})
}

// logical_xor evaluates element truth values with standard tensor broadcasting.
pub fn (a &Tensor[T]) logical_xor[T](b &Tensor[T]) !&Tensor[bool] {
	return logical_binary[T](a, b, fn [T](left T, right T) bool {
		return td[T](left).bool() != td[T](right).bool()
	})
}

// logical_not negates the truth value of each tensor element.
pub fn (t &Tensor[T]) logical_not[T]() &Tensor[bool] {
	return map_predicate[T](t, fn [T](value T) bool {
		return !td[T](value).bool()
	})
}

fn logical_binary[T](a &Tensor[T], b &Tensor[T], operation fn (left T, right T) bool) !&Tensor[bool] {
	mut iterators, shape := a.iterators[T]([b])!
	mut result := empty[bool](shape)
	for {
		values, index := iterators.next() or { break }
		result.set(index, operation(values[0], values[1]))
	}
	return result
}

// all_axis reduces one axis with logical AND. The reduced dimension is removed
// unless keepdims is true. Empty reductions follow NumPy's identity: true.
pub fn (t &Tensor[T]) all_axis[T](axis int, keepdims bool) !&Tensor[bool] {
	return logical_reduce_axis[T](t, axis, keepdims, true)
}

// any_axis reduces one axis with logical OR. The reduced dimension is removed
// unless keepdims is true. Empty reductions follow NumPy's identity: false.
pub fn (t &Tensor[T]) any_axis[T](axis int, keepdims bool) !&Tensor[bool] {
	return logical_reduce_axis[T](t, axis, keepdims, false)
}

// all_axes reduces multiple axes with logical AND. Axes may be negative and
// must be unique. An empty axes list converts values to bool without reducing.
pub fn (t &Tensor[T]) all_axes[T](axes []int, keepdims bool) !&Tensor[bool] {
	return logical_reduce_axes[T](t, axes, keepdims, true)
}

// any_axes reduces multiple axes with logical OR. Axes may be negative and
// must be unique. An empty axes list converts values to bool without reducing.
pub fn (t &Tensor[T]) any_axes[T](axes []int, keepdims bool) !&Tensor[bool] {
	return logical_reduce_axes[T](t, axes, keepdims, false)
}

fn logical_reduce_axes[T](t &Tensor[T], axes []int, keepdims bool, reduce_all bool) !&Tensor[bool] {
	if axes.len == 0 {
		mut result := empty[bool](t.shape, memory: .row_major)
		mut iter := t.iterator[T]()
		for {
			value, index := iter.next() or { break }
			result.set(index, td[T](value).bool())
		}
		return result
	}
	mut normalized := []int{cap: axes.len}
	for axis in axes {
		axis_index := if axis < 0 { axis + t.rank() } else { axis }
		if axis_index < 0 || axis_index >= t.rank() {
			return error('axis ${axis} out of bounds for rank ${t.rank()}')
		}
		if axis_index in normalized {
			return error('duplicate axis ${axis}')
		}
		normalized << axis_index
	}
	// Reduce from the highest axis down so removing dimensions does not change
	// the indices of axes still to process.
	for i in 0 .. normalized.len {
		for j in i + 1 .. normalized.len {
			if normalized[i] < normalized[j] {
				normalized[i], normalized[j] = normalized[j], normalized[i]
			}
		}
	}
	mut result := logical_reduce_axis[T](t, normalized[0], keepdims, reduce_all)!
	for axis_index in normalized[1..] {
		result = logical_reduce_axis[bool](result, axis_index, keepdims, reduce_all)!
	}
	return result
}

fn logical_reduce_axis[T](t &Tensor[T], axis int, keepdims bool, reduce_all bool) !&Tensor[bool] {
	rank := t.rank()
	if rank == 0 {
		return error('logical axis reduction requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('axis ${axis} out of bounds for rank ${rank}')
	}
	mut output_shape := t.shape.clone()
	if keepdims {
		output_shape[axis_index] = 1
	} else {
		output_shape.delete(axis_index)
	}
	mut result := empty[bool](output_shape, memory: .row_major)
	mut slice_count := 1
	for dimension, size in t.shape {
		if dimension != axis_index {
			slice_count *= size
		}
	}
	mut index := []int{len: rank}
	for slice in 0 .. slice_count {
		decode_logical_reduction_slice(slice, t.shape, axis_index, mut index)
		mut reduced := reduce_all
		for position in 0 .. t.shape[axis_index] {
			index[axis_index] = position
			value := td[T](t.get(index)).bool()
			if reduce_all {
				reduced = reduced && value
			} else {
				reduced = reduced || value
			}
			if reduced != reduce_all {
				break
			}
		}
		if keepdims {
			index[axis_index] = 0
			result.set(index, reduced)
		} else {
			mut output_index := index[..axis_index].clone()
			output_index << index[axis_index + 1..]
			result.set(output_index, reduced)
		}
	}
	return result
}

fn decode_logical_reduction_slice(line int, shape []int, axis int, mut index []int) {
	mut remainder := line
	for dimension := shape.len - 1; dimension >= 0; dimension-- {
		if dimension == axis {
			index[dimension] = 0
			continue
		}
		index[dimension] = remainder % shape[dimension]
		remainder /= shape[dimension]
	}
}

// is_finite returns true where x is not positive infinity, negative infinity, or NaN;
// false otherwise.
pub fn (t &Tensor[T]) is_finite[T]() &Tensor[bool] {
	return map_predicate[T](t, fn (value T) bool {
		return math.is_finite(td[T](value).f64())
	})
}

// is_inf reports whether t is an infinity, according to sign.
// If sign > 0, is_inf reports whether t is positive infinity.
// If sign < 0, is_inf reports whether t is negative infinity.
// If sign == 0, is_inf reports whether t is either infinity.
pub fn (t &Tensor[T]) is_inf[T](sign int) &Tensor[bool] {
	return map_predicate[T](t, fn [sign] (value T) bool {
		return math.is_inf(td[T](value).f64(), sign)
	})
}

// is_nan returns true for NaN values and false for finite values and infinities.
pub fn (t &Tensor[T]) is_nan[T]() &Tensor[bool] {
	return map_predicate[T](t, fn (value T) bool {
		return math.is_nan(td[T](value).f64())
	})
}

// map_predicate builds a boolean result with a linear fast path for contiguous
// tensors and stride-aware iteration for views.
@[direct_array_access]
fn map_predicate[T](t &Tensor[T], predicate fn (value T) bool) &Tensor[bool] {
	mut result := empty[bool](t.shape, memory: .row_major)
	if t.is_row_major_contiguous() && t.data.data.len == t.size {
		for i in 0 .. t.size {
			result.data.data[i] = predicate(t.data.data[i])
		}
		return result
	}
	mut iter := t.iterator[T]()
	for {
		value, index := iter.next() or { break }
		result.set(index, predicate(value))
	}
	return result
}

// array_equal returns true if input arrays have the same shape and all elements
// compare exactly equal. Floating-point NaNs compare unequal; use isclose for
// approximate comparisons.
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

@[direct_array_access; inline]
fn handle_equal[T](vals []T, _ []int) bool {
	mut equal := true
	for v in vals {
		equal = equal && v == vals[0]
	}
	return equal
}

// equal compares two tensors elementwise

// equal exposes this operation as part of the public API.

// equal exposes this operation as part of the public API.
@[direct_array_access]
pub fn (t &Tensor[T]) equal[T](other &Tensor[T]) !&Tensor[bool] {
	if t.shape == other.shape && t.is_row_major_contiguous() && other.is_row_major_contiguous()
		&& t.data.data.len == t.size && other.data.data.len == other.size {
		mut ret := empty[bool](t.shape)
		for i in 0 .. t.size {
			ret.data.data[i] = t.data.data[i] == other.data.data[i]
		}
		return ret
	}
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

// not_equal exposes this operation as part of the public API.

// not_equal exposes this operation as part of the public API.
@[direct_array_access]
pub fn (t &Tensor[T]) not_equal[T](other &Tensor[T]) !&Tensor[bool] {
	if t.shape == other.shape && t.is_row_major_contiguous() && other.is_row_major_contiguous()
		&& t.data.data.len == t.size && other.data.data.len == other.size {
		mut ret := empty[bool](t.shape)
		for i in 0 .. t.size {
			ret.data.data[i] = t.data.data[i] != other.data.data[i]
		}
		return ret
	}
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

// tolerance exposes this operation as part of the public API.

// tolerance exposes this operation as part of the public API.
@[inline]
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

// close exposes this operation as part of the public API.

// close exposes this operation as part of the public API.
@[inline]
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

// veryclose exposes this operation as part of the public API.

// veryclose exposes this operation as part of the public API.
@[inline]
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

// alike exposes this operation as part of the public API.

// alike exposes this operation as part of the public API.
@[inline]
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

// isclose compares tensors elementwise using NumPy's asymmetric tolerance rule:
// abs(a - b) <= atol + rtol * abs(b). The returned tensor has the broadcast
// shape of the inputs. NaNs compare false unless equal_nan is true.

// IsCloseData configures relative and absolute tolerances and NaN comparison.
@[params]
pub struct IsCloseData {
pub:
	rtol      f64 = 1e-5
	atol      f64 = 1e-8
	equal_nan bool
}

pub fn (t &Tensor[T]) isclose[T](other &Tensor[T], params IsCloseData) !&Tensor[bool] {
	validate_isclose_tolerances(params)!
	mut iters, shape := t.iterators[T]([other])!
	mut ret := empty[bool](shape)
	for {
		vals, i := iters.next() or { break }
		a := td[T](vals[0]).f64()
		b := td[T](vals[1]).f64()
		ret.set(i, isclose_values(a, b, params.rtol, params.atol, params.equal_nan))
	}
	return ret
}

// allclose returns true if all broadcasted elements satisfy isclose.
@[direct_array_access]
pub fn (t &Tensor[T]) allclose[T](other &Tensor[T], params IsCloseData) !bool {
	validate_isclose_tolerances(params)!
	if t.shape == other.shape && t.is_row_major_contiguous() && other.is_row_major_contiguous()
		&& t.data.data.len == t.size && other.data.data.len == other.size {
		for i in 0 .. t.size {
			a := td[T](t.data.data[i]).f64()
			b := td[T](other.data.data[i]).f64()
			if !isclose_values(a, b, params.rtol, params.atol, params.equal_nan) {
				return false
			}
		}
		return true
	}
	mut iters, _ := t.iterators[T]([other])!
	for {
		vals, _ := iters.next() or { break }
		a := td[T](vals[0]).f64()
		b := td[T](vals[1]).f64()
		if !isclose_values(a, b, params.rtol, params.atol, params.equal_nan) {
			return false
		}
	}
	return true
}

fn validate_isclose_tolerances(params IsCloseData) ! {
	if params.rtol < 0 || params.atol < 0 || math.is_nan(params.rtol) || math.is_nan(params.atol)
		|| math.is_inf(params.rtol, 0) || math.is_inf(params.atol, 0) {
		return error('rtol and atol must be non-negative finite numbers')
	}
}

fn isclose_values(a f64, b f64, rtol f64, atol f64, equal_nan bool) bool {
	if a == b {
		return true
	}
	if math.is_nan(a) || math.is_nan(b) {
		return equal_nan && math.is_nan(a) && math.is_nan(b)
	}
	if math.is_inf(a, 0) || math.is_inf(b, 0) {
		return false
	}
	// Scale before subtracting so finite values near f64's maximum cannot
	// overflow both sides of the comparison and incorrectly compare close.
	scale := math.max(math.max(math.abs(a), math.abs(b)), atol)
	if scale == 0 {
		return true
	}
	return math.abs(a / scale - b / scale) <= atol / scale + rtol * math.abs(b / scale)
}
