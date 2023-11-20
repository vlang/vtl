module vtl

// assert_square_matrix panics if the given tensor is not a square matrix
@[inline]
fn (t &VclTensor[T]) assert_square_matrix[T]() ! {
	if t.is_square_matrix() {
		return error('Matrix is not square')
	}
}

// assert_square_matrix panics if the given tensor is not a matrix
@[inline]
fn (t &VclTensor[T]) assert_matrix[T]() ! {
	if t.is_matrix() {
		return error('Tensor is not two-dimensional')
	}
}

// assert_rank ensures that a Tensor has a given rank
@[inline]
fn (t &VclTensor[T]) assert_rank[T](n int) ! {
	if n != t.rank() {
		return error('Bad number of dimensions')
	}
}

// assert_min_rank ensures that a Tensor has at least a given rank
@[inline]
fn (t &VclTensor[T]) assert_min_rank[T](n int) ! {
	if n > t.rank() {
		return error('Bad number of dimensions')
	}
}

// ensure_memory sets a correct memory layout to a given tensor
@[inline]
pub fn (mut t VclTensor[T]) ensure_memory[T]() {
	if t.is_col_major() {
		if !t.is_col_major_contiguous() {
			t.memory = .row_major
		}
	}
	if t.is_contiguous() {
		if t.rank() > 1 {
			t.memory = .row_major
		}
	}
}
