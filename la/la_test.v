module la

import math
import math.complex as vcomplex
import vtl

fn test_dot_1() {
	a := vtl.from_1d([1.0, 2.0, 3.0])!
	b := vtl.from_1d([4.0, 5.0, 6.0])!
	expected := vtl.from_1d([32.0])!
	result := dot(a, b)!
	assert result.shape == [1]
	assert result.array_equal(expected)
}

fn test_det_1() {
	a := vtl.from_2d([[1.0, 0], [0.0, 1]])!
	expected := vtl.from_1d([1.0])!
	result := det(a)!
	assert result.shape == [1]
	assert result.array_equal(expected)
}

fn test_det_supports_batches_and_singular_matrices() {
	a := vtl.from_array([1.0, 2, 3, 4, 1, 2, 2, 4], [2, 2, 2])!
	result := det(a)!
	assert result.shape == [2]
	assert math.abs(result.get_nth(0) + 2.0) < 1e-12
	assert result.get_nth(1) == 0.0
}

fn test_inv_1() {
	a := vtl.from_2d([[4.0, 7.0], [2.0, 6.0]])!
	result := inv(a)!
	assert result.shape == [2, 2]
	assert math.abs(result.get([0, 0]) - 0.6) < 1e-12
	assert math.abs(result.get([0, 1]) + 0.7) < 1e-12
	assert math.abs(result.get([1, 0]) + 0.2) < 1e-12
	assert math.abs(result.get([1, 1]) - 0.4) < 1e-12
}

fn test_inv_supports_batches() {
	a := vtl.from_array([1.0, 2, 3, 4, 2, 0, 0, 4], [2, 2, 2])!
	result := inv(a)!
	assert result.shape == [2, 2, 2]
	expected := [-2.0, 1.0, 1.5, -0.5, 0.5, 0.0, 0.0, 0.25]
	for i, value in expected {
		assert math.abs(result.get_nth(i) - value) < 1e-12
	}
}

fn test_inv_rejects_singular_matrices_in_a_batch() {
	a := vtl.from_array([1.0, 2, 3, 4, 1, 2, 2, 4], [2, 2, 2])!
	if _ := inv(a) {
		assert false, 'inv must reject a singular matrix in the batch'
	} else {
		assert true
	}
}

fn test_matmul_1() {
	a := vtl.from_2d([[1.0, 0], [0.0, 1]])!
	b := vtl.from_2d([[4.0, 1], [2.0, 2]])!
	expected := vtl.from_2d([[4.0, 1], [2.0, 2]])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.array_equal(expected)
}

fn test_matmul_preserves_integer_dtype_and_values() {
	a := vtl.from_array([1, 2, 3, 4], [2, 2])!
	b := vtl.from_array([5, 6, 7, 8], [2, 2])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.to_array() == [19, 22, 43, 50]
}

fn test_matmul_preserves_f32_dtype() {
	a := vtl.from_array([f32(1), 2, 3, 4], [2, 2])!
	b := vtl.from_array([f32(5), 6, 7, 8], [2, 2])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.to_array() == [f32(19), 22, 43, 50]
}

fn test_matmul_complex128_matrix_product() {
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
	result := matmul(a, b)!
	assert result.dtype() == .complex128
	assert result.shape == [2, 2]
	assert result.to_array() == [
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2, im: 0 },
		vcomplex.Complex{ re: 5, im: 3 },
		vcomplex.Complex{ re: 6, im: -6 },
	]
}

fn test_matmul_complex128_falls_back_when_three_product_overflows() {
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1e154, im: 1e154 },
	], [1, 1])!
	b := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1e154, im: 0 },
	], [1, 1])!
	result := matmul(a, b)!
	value := result.to_array()[0]
	assert math.is_finite(value.re)
	assert math.is_finite(value.im)
	assert value.re == 1e308
	assert value.im == 1e308
}

fn test_matmul_complex128_falls_back_when_small_imag_is_absorbed() {
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 9007199254740992.0, im: 1.0 },
	], [1, 1])!
	b := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1.0, im: 0.0 },
	], [1, 1])!
	result := matmul(a, b)!
	value := result.to_array()[0]
	assert value.re == 9007199254740992.0
	assert value.im == 1.0
}

fn test_matmul_complex128_falls_back_when_im_cancels_below_rounding_bound() {
	// The (Re+Im) sums preserve both operands and every product stays
	// finite, but Im = n*2^-52 is tiny next to the operand-term scale n:
	// the term bound rejects the three-product form. The component-wise
	// cross terms compute Im exactly.
	delta := math.pow(2.0, -52.0)
	n := 768
	mut a_vals := []vcomplex.Complex{len: n}
	for i in 0 .. n {
		a_vals[i] = vcomplex.Complex{ re: delta, im: 1.0 }
	}
	a := vtl.from_array[vcomplex.Complex](a_vals, [1, n])!
	mut b_vals := []vcomplex.Complex{len: n, init: vcomplex.Complex{ re: 0.0, im: 1.0 }}
	b := vtl.from_array[vcomplex.Complex](b_vals, [n, 1])!
	result := matmul(a, b)!
	value := result.to_array()[0]
	assert value.re == -f64(n)
	assert value.im == f64(n) * delta
}

fn test_three_product_accurate_detects_cancellation() {
	// Im = p3-p1-p2 cancels to zero while the operand terms stay full
	// scale: the term bound exceeds any fraction of Im.
	assert !three_product_accurate([1e10], [1e10], [2e10], [1e10])
	// The products can look benign while the terms behind them are huge:
	// the output-only bound accepted this shape with Im off by one or two.
	assert !three_product_accurate([0.0], [4.0], [8.0], [36.0])
	// Re is guarded too: p1-p2 loses a small real part to absorption.
	big := math.pow(2.0, 60.0)
	assert !three_product_accurate([big], [big], [2.0 * big], [8192.0])
	// A well-scaled result passes.
	assert three_product_accurate([1.0], [2.0], [4.0], [1e-10])
	// All-zero products are exact and pass.
	assert three_product_accurate([0.0], [0.0], [0.0], [0.0])
}

fn test_complex_term_bound_tracks_operand_magnitudes() {
	// A=[1+d*i, -1+d*i], B=[M+2M*i, M+2M*i]^T: the products nearly
	// cancel, but the operand terms are of scale 3*M, so the bound must
	// stay far above the quarter-of-Im acceptance threshold.
	d := math.pow(2.0, -52.0)
	m := math.pow(2.0, 52.0)
	bound := complex_term_bound([1.0, -1.0], [d, d], [m, m], [2.0 * m, 2.0 * m], 1, 1, 2)
	assert bound.len == 1
	assert bound[0] > 1.0
	// Doubling every operand scales the row sum and the column maximum,
	// hence the bound, by four.
	scaled := complex_term_bound([2.0, -2.0], [2.0 * d, 2.0 * d], [2.0 * m, 2.0 * m], [
		4.0 * m,
		4.0 * m,
	], 1, 1, 2)
	assert scaled[0] == 4.0 * bound[0]
}

fn test_three_product_results_finite_rejects_non_finite_combination() {
	// p1-p2 overflows even though each product is finite.
	assert !three_product_results_finite([1e308], [-1e308], [0.0])
	assert three_product_results_finite([1.0], [2.0], [4.0])
}

fn test_matmul_complex128_falls_back_when_term_rounding_dominates_im() {
	// Review counterexample: d=2^-52, M=2^52. The operand sums are exact,
	// p1=0, p2=4, but p3 rounds to 8 (or 7 with FMA) instead of 6, so
	// Im = 4 (or 3) instead of 2. The term bound rejects the three-product
	// form; the cross terms give the exact result.
	d := math.pow(2.0, -52.0)
	m := math.pow(2.0, 52.0)
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1.0, im: d },
		vcomplex.Complex{ re: -1.0, im: d },
	], [1, 2])!
	b := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: m, im: 2.0 * m },
		vcomplex.Complex{ re: m, im: 2.0 * m },
	], [2, 1])!
	result := matmul(a, b)!
	value := result.to_array()[0]
	assert value.re == -4.0
	assert value.im == 2.0
}

fn test_matmul_complex128_rejects_three_product_with_cancelled_operands() {
	// Maintainer review: A=(2^53+2i), B=(2^53-2i). Both sums preserve
	// their operands; p1=p3=2^106 and p2=-4, so the three-product Im
	// comes back as 4 while the exact cross-term Im is 0.
	big := math.pow(2.0, 53.0)
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: big, im: 2.0 },
	], [1, 1])!
	b := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: big, im: -2.0 },
	], [1, 1])!
	result := matmul(a, b)!
	value := result.to_array()[0]
	assert value.re == math.pow(2.0, 106.0)
	assert value.im == 0.0
}

fn test_matmul_complex128_avoids_nan_when_re_products_overflow() {
	// x=2^511: x*x=2^1022 is finite, but four such terms overflow both
	// p1 and p2 to Inf. Reusing them gives Inf-Inf=NaN for a result
	// whose exact real part is 0; fall back to term-by-term cancellation.
	x := math.pow(2.0, 511.0)
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: x, im: x },
		vcomplex.Complex{ re: x, im: -x },
		vcomplex.Complex{ re: x, im: x },
		vcomplex.Complex{ re: x, im: -x },
	], [1, 4])!
	b := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: x, im: x },
		vcomplex.Complex{ re: x, im: -x },
		vcomplex.Complex{ re: x, im: x },
		vcomplex.Complex{ re: x, im: -x },
	], [4, 1])!
	result := matmul(a, b)!
	value := result.to_array()[0]
	assert value.re == 0.0
	assert value.im == 0.0
}

fn test_matmul_complex128_keeps_real_part_when_products_absorb_it() {
	// Real part 100 is below half-ulp of 2^60: p1 and p2 both round to
	// 2^60, so p1-p2 is zero. The term-wise form keeps the 100.
	big := math.pow(2.0, 60.0)
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: big, im: big },
		vcomplex.Complex{ re: 100.0, im: 0.0 },
	], [1, 2])!
	b := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1.0, im: 1.0 },
		vcomplex.Complex{ re: 1.0, im: 0.0 },
	], [2, 1])!
	result := matmul(a, b)!
	value := result.to_array()[0]
	assert value.re == 100.0
	assert value.im == 2.0 * big
}

fn test_matmul_complex128_fallback_keeps_small_im_beside_large_cross_terms() {
	// The cross terms are 2^60 and -2^60, and the exact Im is 1. Two
	// separately summed cross products absorb the 1 into 2^60; the
	// term-wise form cancels 2^60 first and keeps 1.
	big := math.pow(2.0, 60.0)
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: big, im: 0.0 },
		vcomplex.Complex{ re: 0.0, im: -big },
		vcomplex.Complex{ re: 1.0, im: 0.0 },
	], [1, 3])!
	b := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 0.0, im: 1.0 },
		vcomplex.Complex{ re: 1.0, im: 0.0 },
		vcomplex.Complex{ re: 0.0, im: 1.0 },
	], [3, 1])!
	result := matmul(a, b)!
	value := result.to_array()[0]
	assert value.re == 0.0
	assert value.im == 1.0
}

fn test_matmul_complex128_rejects_im_with_large_rounding_error() {
	// The old quarter-of-Im threshold accepted this input with ~1% Im
	// error; an error bound of at most 1e-8 of the result sends it
	// term-wise, where the cross terms are exact.
	m := math.pow(2.0, 53.0)
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 100.0, im: m },
	], [1, 1])!
	b := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1.0, im: -m },
	], [1, 1])!
	result := matmul(a, b)!
	value := result.to_array()[0]
	assert value.re == math.pow(2.0, 106.0)
	assert value.im == -99.0 * m
}

fn test_matmul_complex128_fallback_after_three_product_batch() {
	// A benign batch accepted by the three-product path must not carry a
	// later batch that needs the fallback: used_three_product is per batch.
	d := math.pow(2.0, -52.0)
	m := math.pow(2.0, 52.0)
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 1, im: d },
		vcomplex.Complex{ re: -1, im: d },
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 1, im: 1 },
	], [2, 2, 2])!
	big_col := []vcomplex.Complex{len: 4, init: vcomplex.Complex{ re: m, im: 2.0 * m }}
	b := vtl.from_array[vcomplex.Complex](big_col, [2, 2, 1])!
	result := matmul(a, b)!
	values := result.to_array()
	benign := vcomplex.Complex{ re: -2.0 * m, im: 6.0 * m }
	assert values[0] == benign
	assert values[1] == benign
	assert values[2] == vcomplex.Complex{ re: -4.0, im: 2.0 }
	assert values[3] == benign
}

fn test_matmul_complex128_fallback_checks_every_entry() {
	// Row 0 is accepted by the three-product path; row 1 needs the
	// fallback. Checking only the first entry would return Im=4 for row 1.
	d := math.pow(2.0, -52.0)
	m := math.pow(2.0, 52.0)
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 1, im: d },
		vcomplex.Complex{ re: -1, im: d },
	], [2, 2])!
	b := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: m, im: 2.0 * m },
		vcomplex.Complex{ re: m, im: 2.0 * m },
	], [2, 1])!
	result := matmul(a, b)!
	values := result.to_array()
	assert values[0] == vcomplex.Complex{ re: -2.0 * m, im: 6.0 * m }
	assert values[1] == vcomplex.Complex{ re: -4.0, im: 2.0 }
}

fn test_matmul_complex128_zero_sized_dimensions() {
	// Complex matmul has its own zero-size early return and inner==0
	// continue branch; neither is covered by the f64 zero-size test.
	a := vtl.from_array([]vcomplex.Complex{}, [2, 0, 4])!
	b := vtl.from_array([]vcomplex.Complex{len: 2 * 4 * 3, init: vcomplex.Complex{ re: 1, im: 0 }}, [
		2,
		4,
		3,
	])!
	result := matmul(a, b)!
	assert result.shape == [2, 0, 3]
	assert result.size == 0

	c := vtl.from_array([]vcomplex.Complex{}, [2, 3, 0])!
	d := vtl.from_array([]vcomplex.Complex{}, [2, 0, 4])!
	empty_inner := matmul(c, d)!
	assert empty_inner.shape == [2, 3, 4]
	assert empty_inner.to_array() == []vcomplex.Complex{len: 2 * 3 * 4, init: vcomplex.Complex{}}
}

fn test_matmul_complex128_vector_dot_does_not_conjugate() {
	a := vtl.from_1d[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2, im: 0 },
	])!
	b := vtl.from_1d[vcomplex.Complex]([
		vcomplex.Complex{ re: 0, im: 1 },
		vcomplex.Complex{ re: 1, im: 0 },
	])!
	result := matmul(a, b)!
	assert result.rank() == 0
	assert result.get_nth[vcomplex.Complex](0) == vcomplex.Complex{ re: 1, im: 1 }
}

fn test_matmul_broadcasts_complex128_batch_dimensions() {
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2, im: 0 },
	], [1, 1, 2])!
	b := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 0, im: 1 },
		vcomplex.Complex{ re: 1, im: 0 },
		vcomplex.Complex{ re: 2, im: 0 },
		vcomplex.Complex{ re: 0, im: -1 },
	], [2, 2, 1])!
	result := matmul(a, b)!
	assert result.shape == [2, 1, 1]
	assert result.to_array() == [vcomplex.Complex{ re: 1, im: 1 }, vcomplex.Complex{ re: 2, im: 0 }]
}

fn test_matmul_complex128_reads_non_contiguous_views() {
	input := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 0 },
		vcomplex.Complex{ re: 2, im: 0 },
		vcomplex.Complex{ re: 3, im: 0 },
		vcomplex.Complex{ re: 4, im: 0 },
	], [2, 2])!
	transposed := input.transpose([1, 0])!
	identity := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 0 },
		vcomplex.Complex{ re: 0, im: 0 },
		vcomplex.Complex{ re: 0, im: 0 },
		vcomplex.Complex{ re: 1, im: 0 },
	], [2, 2])!
	result := matmul(transposed, identity)!
	assert result.to_array() == [
		vcomplex.Complex{ re: 1, im: 0 },
		vcomplex.Complex{ re: 3, im: 0 },
		vcomplex.Complex{ re: 2, im: 0 },
		vcomplex.Complex{ re: 4, im: 0 },
	]
}

fn test_matmul_f32_rectangular_matrix() {
	a := vtl.from_array([f32(1), 2, 3, 4, 5, 6], [2, 3])!
	b := vtl.from_array([f32(7), 8, 9, 10, 11, 12], [3, 2])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.to_array() == [f32(58), 64, 139, 154]
}

fn test_matmul_broadcasts_f32_batch() {
	a := vtl.from_array([f32(1), 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	b := vtl.from_array([f32(1), 0, 0, 1], [2, 2])!
	result := matmul(a, b)!
	assert result.shape == [2, 2, 2]
	assert result.to_array() == [f32(1), 2, 3, 4, 5, 6, 7, 8]
}

fn test_matmul_2() {
	a := vtl.seq[f64](2 * 2 * 4).reshape([2, 2, 4])!
	b := vtl.seq[f64](2 * 2 * 4).reshape([2, 4, 2])!
	result := matmul(a, b)!
	assert result.shape == [2, 2, 2]
	expected := vtl.from_array([28.0, 34, 76, 98, 428, 466, 604, 658], [2, 2, 2])!
	assert result.array_equal(expected)
}

fn test_matmul_broadcasts_batch_dimensions() {
	a := vtl.seq[f64](8).reshape([2, 1, 2, 2])!
	b := vtl.from_array([f64(1), 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [3, 2, 2])!
	result := matmul(a, b)!
	assert result.shape == [2, 3, 2, 2]
	expected := vtl.from_array([f64(0), 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7,
		4, 5, 6, 7], [2, 3, 2, 2])!
	assert result.array_equal(expected)
}

fn test_matmul_rejects_incompatible_batch_dimensions() {
	a := vtl.ones[f64]([2, 2, 3])
	b := vtl.ones[f64]([3, 3, 2])
	if _ := matmul(a, b) {
		assert false
	} else {
		assert true
	}
}

fn test_matmul_vector_vector_returns_scalar() {
	a := vtl.from_1d([1.0, 2, 3])!
	b := vtl.from_1d([4.0, 5, 6])!
	result := matmul(a, b)!
	assert result.rank() == 0
	assert result.get_nth[f64](0) == 32.0
}

fn test_matmul_matrix_vector_returns_vector() {
	a := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	b := vtl.from_1d([5.0, 6.0])!
	result := matmul(a, b)!
	assert result.shape == [2]
	assert result.to_array() == [17.0, 39.0]
}

fn test_matmul_vector_matrix_returns_vector() {
	a := vtl.from_1d([1.0, 2.0])!
	b := vtl.from_2d([[3.0, 4.0], [5.0, 6.0]])!
	result := matmul(a, b)!
	assert result.shape == [2]
	assert result.to_array() == [13.0, 16.0]
}

fn test_matmul_batched_matrix_vector_returns_batched_vectors() {
	a := vtl.seq[f64](12).reshape([2, 2, 3])!
	b := vtl.from_1d([1.0, 0, -1])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.to_array() == [-2.0, -2.0, -2.0, -2.0]
}

fn test_matmul_rejects_scalar_operands() {
	a := vtl.from_array([2.0], [])!
	b := vtl.from_1d([1.0, 2])!
	if _ := matmul(a, b) {
		assert false
	} else {
		assert true
	}
}

fn test_matmul_zero_sized_dimensions() {
	a := vtl.from_array([]f64{}, [2, 0, 4])!
	b := vtl.seq[f64](2 * 4 * 3).reshape([2, 4, 3])!
	result := matmul(a, b)!
	assert result.shape == [2, 0, 3]
	assert result.size == 0

	c := vtl.from_array([]f64{}, [2, 3, 0])!
	d := vtl.from_array([]f64{}, [2, 0, 4])!
	expected := vtl.zeros[f64]([2, 3, 4])
	result_with_empty_inner := matmul(c, d)!
	assert result_with_empty_inner.shape == [2, 3, 4]
	assert result_with_empty_inner.array_equal(expected)
}

fn test_matmul_column_major_operand() {
	a := vtl.from_array([1.0, 2, 3, 4], [1, 2, 2], memory: .col_major)!
	b := vtl.from_array([1.0, 0, 0, 1], [1, 2, 2])!
	result := matmul(a, b)!
	expected := vtl.from_array([1.0, 2, 3, 4], [1, 2, 2], memory: .col_major)!
	assert result.array_equal(expected)
}

fn test_matmul_non_contiguous_rank_two_operands() {
	a := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!.transpose([1, 0])!
	b := vtl.from_2d([[1.0, 0.0], [0.0, 1.0]])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.to_array() == [1.0, 3.0, 2.0, 4.0]
}
