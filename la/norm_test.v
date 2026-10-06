module la

import math
import vtl

fn test_vector_norm_supports_common_orders() ! {
	values := vtl.from_1d([3.0, -4.0, 0.0])!
	assert vector_norm(values, 1)!.get_nth(0) == 7.0
	assert vector_norm(values, 2)!.get_nth(0) == 5.0
	assert vector_norm(values, 0)!.get_nth(0) == 2.0
	assert vector_norm(values, math.inf(1))!.get_nth(0) == 4.0
	assert vector_norm(values, math.inf(-1))!.get_nth(0) == 0.0
	assert vector_norm(values, -1)!.get_nth(0) == 0.0
	positive_values := vtl.from_1d([3.0, 4.0])!
	assert math.abs(vector_norm(positive_values, -1)!.get_nth(0) - (12.0 / 7.0)) < 1e-12
	assert math.abs(vector_norm(positive_values, 3)!.get_nth(0) - math.pow(91.0, 1.0 / 3.0)) < 1e-12
}

fn test_vector_norm_axis_retains_dimensions_for_strided_views() ! {
	values := vtl.from_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!.transpose([1, 0])!
	rows := vector_norm_axis(values, 2, 1)!
	assert rows.shape == [3]
	assert math.abs(rows.get([0]) - math.sqrt(17.0)) < 1e-12
	assert math.abs(rows.get([1]) - math.sqrt(29.0)) < 1e-12
	assert math.abs(rows.get([2]) - math.sqrt(45.0)) < 1e-12
	rows_keepdims := vector_norm_axis_keepdims(values, 2, 1)!
	assert rows_keepdims.shape == [3, 1]
	columns := vector_norm_axis(values, 2, -2)!
	assert columns.shape == [2]
	assert math.abs(columns.get([0]) - math.sqrt(14.0)) < 1e-12
	assert math.abs(columns.get([1]) - math.sqrt(77.0)) < 1e-12
	vector := vtl.from_1d([3.0, 4.0])!
	scalar := vector_norm_axis(vector, 2, 0)!
	assert scalar.rank() == 0
	assert scalar.get([]) == 5.0
}

fn test_vector_norm_axes_reduces_axis_tuples_and_keeps_dimensions() ! {
	values := vtl.from_3d([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])!
	got := vector_norm_axes(values, 2, [0, 2], false)!
	assert got.shape == [2]
	assert math.abs(got.get([0]) - math.sqrt(66.0)) < 1e-12
	assert math.abs(got.get([1]) - math.sqrt(138.0)) < 1e-12
	kept := vector_norm_axes(values, 2, [-1, 0], true)!
	assert kept.shape == [1, 2, 1]
	assert math.abs(kept.get([0, 0, 0]) - math.sqrt(66.0)) < 1e-12
	assert math.abs(kept.get([0, 1, 0]) - math.sqrt(138.0)) < 1e-12
	transposed := values.transpose([2, 1, 0])!
	strided := vector_norm_axes(transposed, 2, [0, 2], false)!
	assert strided.shape == [2]
	assert math.abs(strided.get([0]) - math.sqrt(66.0)) < 1e-12
	assert math.abs(strided.get([1]) - math.sqrt(138.0)) < 1e-12
}

fn test_vector_norm_axes_rejects_invalid_and_duplicate_axes() ! {
	values := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	if _ := vector_norm_axes(values, 2, [0, -2], false) {
		assert false, 'duplicate normalized axes must be rejected'
	}
	if _ := vector_norm_axes(values, 2, [2], false) {
		assert false, 'out-of-range axes must be rejected'
	}
	if _ := vector_norm_axes(values, 2, [], false) {
		assert false, 'empty axes must be rejected'
	}
}

fn test_vector_norm_scales_large_finite_values() ! {
	values := vtl.from_1d([1e308, 1e308])!
	got := vector_norm(values, 2)!.get_nth(0)
	assert math.abs(got / 1e308 - math.sqrt(2.0)) < 1e-12
}

fn test_vector_norm_empty_semantics_and_invalid_order_or_axis() {
	empty := vtl.from_1d([]f64{})!
	assert vector_norm(empty, 2)!.get_nth(0) == 0.0
	assert vector_norm(empty, 0)!.get_nth(0) == 0.0
	if _ := vector_norm(empty, -1) {
		assert false, 'negative order must reject empty tensors'
	}
	if _ := vector_norm(empty, math.inf(1)) {
		assert false, 'infinite order must reject empty tensors'
	}
	values := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	if _ := vector_norm(values, math.nan()) {
		assert false, 'vector_norm must reject a NaN order'
	}
	if _ := vector_norm_axis(values, 2, 2) {
		assert false, 'vector_norm_axis must reject an invalid axis'
	}
	empty_axis := vtl.empty[f64]([2, 0])
	assert vector_norm_axis(empty_axis, 2, 1)!.to_array() == [0.0, 0.0]
	if _ := vector_norm_axis(empty_axis, -1, 1) {
		assert false, 'negative order must reject an empty reduction axis'
	}
}

fn test_vector_norm_zero_order_counts_nan_as_nonzero() ! {
	values := vtl.from_1d([0.0, math.nan(), 2.0])!
	assert vector_norm(values, 0)!.get_nth(0) == 2.0
	assert math.is_nan(vector_norm(values, 2)!.get_nth(0))
}
