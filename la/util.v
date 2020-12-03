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
