module core

import vtl
import math.complex as vcomplex

fn test_einsum_matrix_multiplication_explicit_and_implicit() ! {
	a := vtl.from_2d([[f64(1), 2], [3, 4]])!
	b := vtl.from_2d([[f64(5), 6], [7, 8]])!

	result := vtl.einsum[f64]('ij,jk->ik', a, b)!
	implicit := vtl.einsum[f64]('ij,jk', a, b)!
	expected := vtl.from_2d([[f64(19), 22], [43, 50]])!
	assert result.array_equal(expected)
	assert implicit.array_equal(expected)
}

fn test_einsum_float32_matrix_product_uses_typed_results() ! {
	a := vtl.from_2d([[f32(1), 2], [3, 4]])!
	b := vtl.from_2d([[f32(2), 0], [0, 2]])!

	result := vtl.einsum[f32]('ij,jk->ik', a, b)!
	assert result.shape == [2, 2]
	assert result.get_nth(0) == f32(2)
	assert result.get_nth(1) == f32(4)
	assert result.get_nth(2) == f32(6)
	assert result.get_nth(3) == f32(8)
}

fn test_einsum_complex128_matrix_product_matches_matmul() ! {
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2, im: 0 },
		vcomplex.Complex{ re: 3, im: -1 },
		vcomplex.Complex{ re: 4, im: 0 },
	], [2, 2])!
	b := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 0, im: 1 },
		vcomplex.Complex{ re: 2, im: 0 },
		vcomplex.Complex{ re: 1, im: 0 },
		vcomplex.Complex{ re: 0, im: -1 },
	], [2, 2])!
	result := vtl.einsum[vcomplex.Complex]('ij,jk->ik', a, b)!
	assert result.shape == [2, 2]
	assert result.to_array() == [
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2, im: 0 },
		vcomplex.Complex{ re: 5, im: 3 },
		vcomplex.Complex{ re: 6, im: -6 },
	]
}

fn test_einsum_batched_matrix_multiplication_with_ellipsis() ! {
	a := vtl.from_array([f64(1), 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	b := vtl.from_array([f64(1), 0, 0, 1, 2, 0, 0, 2], [2, 2, 2])!

	result := vtl.einsum[f64]('...ij,...jk->...ik', a, b)!
	implicit := vtl.einsum[f64]('...ij,...jk', a, b)!
	assert result.shape == [2, 2, 2]
	assert result.get_nth(0) == 1
	assert result.get_nth(3) == 4
	assert result.get_nth(4) == 10
	assert result.get_nth(7) == 16
	assert implicit.array_equal(result)
}

fn test_einsum_ellipsis_broadcasts_batch_axes() ! {
	a := vtl.from_array([f64(1), 2, 3, 4], [1, 2, 2])!
	b := vtl.from_array([f64(1), 0, 0, 1, 2, 0, 0, 2], [2, 2, 2])!

	result := vtl.einsum[f64]('...ij,...jk->...ik', a, b)!
	assert result.shape == [2, 2, 2]
	assert result.get_nth(0) == 1
	assert result.get_nth(3) == 4
	assert result.get_nth(4) == 2
	assert result.get_nth(7) == 8
}

fn test_einsum_ellipsis_can_be_reordered_in_output() ! {
	a := vtl.from_array([f64(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3])!

	result := vtl.einsum[f64]('i...->...i', a)!
	assert result.shape == [2, 3, 2]
	assert result.get_nth(0) == 1
	assert result.get_nth(1) == 7
	assert result.get_nth(2) == 2
	assert result.get_nth(3) == 8
	assert result.get_nth(10) == 6
	assert result.get_nth(11) == 12
}

fn test_einsum_trace_and_diagonal() ! {
	a := vtl.from_2d([[f64(1), 2, 3], [4, 5, 6], [7, 8, 9]])!

	trace := vtl.einsum[f64]('ii->', a)!
	diagonal := vtl.einsum[f64]('ii->i', a)!
	assert trace.rank() == 0
	assert trace.get_nth(0) == 15
	assert diagonal.shape == [3]
	assert diagonal.get_nth(0) == 1
	assert diagonal.get_nth(1) == 5
	assert diagonal.get_nth(2) == 9
}

fn test_einsum_reads_transposed_views() ! {
	a := vtl.from_2d([[f64(1), 2, 3], [4, 5, 6]])!
	transposed_view := a.transpose([1, 0])!

	result := vtl.einsum[f64]('ij->ji', a)!
	assert result.array_equal(transposed_view)
}

fn test_einsum_outer_product_and_multiple_operands() ! {
	a := vtl.from_1d([f64(1), 2])!
	b := vtl.from_1d([f64(3), 4])!
	c := vtl.from_1d([f64(5), 6])!

	outer := vtl.einsum[f64]('i,j->ij', a, b)!
	triple := vtl.einsum[f64]('i,j,k->ijk', a, b, c)!
	assert outer.shape == [2, 2]
	assert outer.get_nth(0) == 3
	assert outer.get_nth(1) == 4
	assert outer.get_nth(2) == 6
	assert outer.get_nth(3) == 8
	assert triple.shape == [2, 2, 2]
	assert triple.get_nth(0) == 15
	assert triple.get_nth(7) == 48
}

fn test_einsum_scalar_operand() ! {
	scalar := vtl.from_array([f64(2)], [])!
	vector := vtl.from_1d([f64(3), 4])!

	result := vtl.einsum[f64](',i->i', scalar, vector)!
	assert result.shape == [2]
	assert result.get_nth(0) == 6
	assert result.get_nth(1) == 8
}

fn test_einsum_broadcasts_size_one_axes() ! {
	a := vtl.from_array([f64(2), 3], [2, 1])!
	b := vtl.from_2d([[f64(10), 20, 30]])!

	result := vtl.einsum[f64]('ij,ij->ij', a, b)!
	assert result.shape == [2, 3]
	assert result.get_nth(0) == 20
	assert result.get_nth(2) == 60
	assert result.get_nth(3) == 30
	assert result.get_nth(5) == 90
}

fn test_einsum_zero_sized_contraction_returns_zeros() ! {
	a := vtl.zeros[f64]([2, 0])
	b := vtl.zeros[f64]([0, 3])

	result := vtl.einsum[f64]('ij,jk->ik', a, b)!
	assert result.shape == [2, 3]
	assert result.get_nth(0) == 0
	assert result.get_nth(5) == 0
}

fn test_einsum_rejects_invalid_expressions() ! {
	a := vtl.ones[f64]([2, 3])
	b := vtl.ones[f64]([4, 2])

	_ := vtl.einsum[f64]('ij,jk->ik', a, b) or {
		assert err.msg().contains('incompatible dimensions')
		return
	}
	assert false, 'expected incompatible dimensions to fail'
}

fn test_einsum_rejects_rank_mismatch() ! {
	a := vtl.ones[f64]([2, 3])

	_ := vtl.einsum[f64]('i', a) or {
		assert err.msg().contains('rank')
		return
	}
	assert false, 'expected input rank mismatch to fail'
}

fn test_einsum_rejects_unknown_output_labels() ! {
	a := vtl.ones[f64]([2, 3])
	_ := vtl.einsum[f64]('ij->ik', a) or {
		assert err.msg().contains('does not appear in an input')
		return
	}
	assert false, 'expected unknown output label to fail'
}

fn test_einsum_preserves_integer_dtype_and_exact_values() ! {
	a := vtl.from_1d([i64(9007199254740993), 1])!
	b := vtl.from_1d([i64(1), 2])!

	result := vtl.einsum[i64]('i,i->', a, b)!
	assert result.get_nth(0) == i64(9007199254740995)
}
