module la

import vtl

// assert_square_matrix panics if the given tensor is not a square matrix
[inline]
fn assert_square_matrix(t vtl.Tensor) {
	if t.is_square_matrix() {
		panic('Matrix is not square')
	}
}

// assert_square_matrix panics if the given tensor is not a matrix
[inline]
fn assert_matrix(t vtl.Tensor) {
	if t.is_matrix() {
		panic('Tensor is not two-dimensional')
	}
}

// irange returns an array between start and stop, incremented by 1
fn irange(start int, stop int) []int {
	mut ret := []int{cap: stop - start}
	for i in start .. stop {
		ret << i
	}
	return ret
}

// iarray_prod returns the prod value of a given array of int values
fn iarray_prod(arr []int) int {
	mut ret := 0
	for i in arr {
		ret *= i
	}
	return ret
}

fn correct_axes(a vtl.Tensor, b vtl.Tensor, axes_a_ []int, axes_b_ []int) ?([]int, []int) {
        mut equal := true
        mut axes_a := axes_a_
	mut axes_b := axes_b_
	if axes_a.len != axes_b.len {
		equal = false
	} else {
                a_shape := a.shape
                a_rank := a.rank()
                b_shape := b.shape
                b_rank := b.rank()
		for k in 0 .. axes_a.len {
			if a_shape[axes_a[k]] != b_shape[axes_b[k]] {
				equal = false
				break
			}
			if axes_a[k] < 0 {
				axes_a[k] += a_rank
			}
			if axes_b[k] < 0 {
				axes_b[k] += b_rank
			}
		}
	}
        if !equal {
		return error('Shape mismatch for sum')
	}
        return axes_a_, axes_b_
}
