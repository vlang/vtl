module num

import vsl.math

// sum returns the sum of all elements of an ndarray
pub fn sum(n NdArray) f64 {
	mut i := 0.0
	for iter := n.iter(); !iter.done; iter.next() {
		unsafe{i += *iter.ptr}
	}
	return i
}

// prod returns the product of an ndarray
pub fn prod(n NdArray) f64 {
	mut i := 1.0
	for iter := n.iter(); !iter.done; iter.next() {
		unsafe{i *= *iter.ptr}
	}
	return i
}

// sum_axis returns the sum of an ndarray along a provided
// axis
pub fn sum_axis(n NdArray, axis int) NdArray {
	return axis_op(n, axis, num_add)
}

// sum_axis_dims returns the sum of an ndarray along a provided
// axis with the reduced dimension intact
pub fn sum_axis_dims(n NdArray, axis int) NdArray {
	return axis_op_dims(n, axis, num_add)
}

// prod_axis returns the product of an ndarray along a provided
// axis
pub fn prod_axis(n NdArray, axis int) NdArray {
	return axis_op(n, axis, num_multiply)
}

// prod_axis_dims returns the product of an ndarray along a provided
// axis with the reduced dimension intact
pub fn prod_axis_dims(n NdArray, axis int) NdArray {
	return axis_op_dims(n, axis, num_multiply)
}

// max returns the maximum value in an ndarray
pub fn max(n NdArray) f64 {
	mut tmp := -math.max_f64
	for iter := n.iter(); !iter.done; iter.next() {
		if *iter.ptr > tmp {
			tmp = *iter.ptr
		}
	}
	return tmp
}

// max_axis returns the maximum value of an ndarray along a
// provided axis
pub fn max_axis(n NdArray, axis int) NdArray {
	return axis_op(n, axis, math.max)
}

// max_axis_dims returns the maximum value of an ndarray along a
// provided axis with the reduced dimension intact
pub fn max_axis_dims(n NdArray, axis int) NdArray {
	return axis_op_dims(n, axis, math.max)
}

// min returns the minimum value in an ndarray
pub fn min(n NdArray) f64 {
	mut tmp := math.max_f64
	for iter := n.iter(); !iter.done; iter.next() {
		if *iter.ptr < tmp {
			tmp = *iter.ptr
		}
	}
	return tmp
}

// min_axis returns the minimum value of an ndarray along a
// provided axis
pub fn min_axis(n NdArray, axis int) NdArray {
	return axis_op(n, axis, math.min)
}

// min_axis_dims returns the minimum value of an ndarray along a
// provided axis with the reduced dimension intact
pub fn min_axis_dims(n NdArray, axis int) NdArray {
	return axis_op_dims(n, axis, math.min)
}

// mean returns the average value of an ndarray
pub fn mean(n NdArray) f64 {
	return sum(n) / n.size
}

// mean_axis returns the average value of an ndarray along a given axis
pub fn mean_axis(n NdArray, axis int) NdArray {
	return divide_as(sum_axis(n, axis), n.shape[axis])
}

// mean_axis_dims returns the average value of an ndarray along a given axis
// with the reduced dimension intact
pub fn mean_axis_dims(n NdArray, axis int) NdArray {
	return divide_as(sum_axis_dims(n, axis), n.shape[axis])
}

// ptp returns the difference between the max and min of an ndarray
pub fn ptp(n NdArray) f64 {
	return max(n) - min(n)
}

// ptp_axis returns the difference between the max and min along an axis
pub fn ptp_axis(n NdArray, axis int) NdArray {
	return subtract(max_axis(n, axis), min_axis(n, axis))
}

// ptp_axis_dims returns the difference between the max and min along an axis
// with the reduced dimension intact
pub fn ptp_axis_dims(n NdArray, axis int) NdArray {
	return subtract(max_axis_dims(n, axis), min_axis_dims(n, axis))
}
