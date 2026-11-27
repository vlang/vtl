module stats

import math
import vtl

fn test_nan_sum_product_and_extrema_ignore_nan() ! {
	values := vtl.from_1d([2.0, math.nan(), -3.0, 4.0])!
	assert nansum(values) == 3.0
	assert nanprod(values) == -24.0
	assert nanmin(values) == -3.0
	assert nanmax(values) == 4.0
	all_nan := vtl.from_1d([math.nan(), math.nan()])!
	assert nansum(all_nan) == 0.0
	assert nanprod(all_nan) == 1.0
	assert math.is_nan(nanmin(all_nan))
	assert math.is_nan(nanmax(all_nan))
	empty := vtl.from_1d([]f64{})!
	assert nansum(empty) == 0.0
	assert nanprod(empty) == 1.0
	assert math.is_nan(nanmin(empty))
	assert math.is_nan(nanmax(empty))
}

fn test_nan_sum_product_global_keepdims_and_typed_accumulators() ! {
	values := vtl.from_array([f32(2), f32(math.nan()), f32(3)], [1, 3])!
	sum_kept := nansum_with_keepdims(values, true)!
	product_kept := nanprod_with_keepdims(values, true)!
	assert sum_kept.shape == [1, 1]
	assert sum_kept.get_nth(0) == 5.0
	assert product_kept.shape == [1, 1]
	assert product_kept.get_nth(0) == 6.0
	assert nansum_with_keepdims(values, false)!.shape == []int{}
	assert nanprod_as_with_keepdims[f32, f64](values, true)!.get_nth(0) == 6.0
	assert nansum_as_with_keepdims[f32, f64](vtl.from_1d([f32(math.nan())])!, true)!.get_nth(0) == 0.0

	empty := vtl.empty[f32]([2, 0])
	assert nansum_with_keepdims(empty, true)!.shape == [1, 1]
	assert nanprod_with_keepdims(empty, true)!.get_nth(0) == 1.0

	scalar_nan := vtl.from_array[f32]([f32(math.nan())], []int{})!
	assert nansum_with_keepdims(scalar_nan, true)!.get_nth(0) == 0.0
	assert nanprod_with_keepdims(scalar_nan, true)!.get_nth(0) == 1.0
	assert nansum_as_with_keepdims[f32, f64](scalar_nan, true)!.get_nth(0) == 0.0
	assert nanprod_as_with_keepdims[f32, f64](scalar_nan, false)!.get_nth(0) == 1.0
}

fn test_nan_axis_sum_product_and_extrema_shapes_and_empty_slices() ! {
	values := vtl.from_array([1.0, math.nan(), 3.0, 4.0, 5.0, 6.0], [2, 3])!
	assert nansum_axis(values, 1)!.to_array() == [4.0, 15.0]
	assert nanprod_axis(values, -1)!.to_array() == [3.0, 120.0]
	assert nanmin_axis(values, 0)!.to_array() == [1.0, 5.0, 3.0]
	assert nanmax_axis(values, 0)!.to_array() == [4.0, 5.0, 6.0]
	assert nansum_axis_keepdims(values, 1)!.shape == [2, 1]
	assert nanprod_axis_keepdims(values, 0)!.shape == [1, 3]
	assert nanmin_axis_keepdims(values, -1)!.shape == [2, 1]
	assert nanmax_axis_keepdims(values, 0)!.shape == [1, 3]
	minimum_without_axis := nanmin_axis_with_keepdims(values, 1, false)!
	assert minimum_without_axis.shape == [2]
	assert minimum_without_axis.to_array() == [1.0, 4.0]
	maximum_with_axis := nanmax_axis_with_keepdims(values, 1, true)!
	assert maximum_with_axis.shape == [2, 1]
	assert maximum_with_axis.to_array() == [3.0, 6.0]

	missing := vtl.from_array([math.nan(), 2.0, math.nan(), math.nan()], [2, 2])!
	sums := nansum_axis(missing, 1)!
	products := nanprod_axis(missing, 1)!
	mins := nanmin_axis(missing, 1)!
	maxs := nanmax_axis(missing, 1)!
	assert sums.to_array() == [2.0, 0.0]
	assert products.to_array() == [2.0, 1.0]
	assert mins.get([0]) == 2.0
	assert math.is_nan(mins.get([1]))
	assert maxs.get([0]) == 2.0
	assert math.is_nan(maxs.get([1]))

	empty_axis := vtl.empty[f64]([2, 0])
	assert nansum_axis(empty_axis, 1)!.to_array() == [0.0, 0.0]
	assert nanprod_axis(empty_axis, 1)!.to_array() == [1.0, 1.0]
	assert math.is_nan(nanmin_axis(empty_axis, 1)!.get([0]))
	assert math.is_nan(nanmax_axis(empty_axis, 1)!.get([1]))
}

fn test_nan_axis_reductions_reject_scalar_and_invalid_axis() ! {
	scalar := vtl.empty[f64]([], memory: .row_major)
	scalar.set([], 1.0)
	if _ := nansum_axis(scalar, 0) {
		assert false, 'axis reduction must reject scalar inputs'
	}
	values := vtl.from_2d([[1.0, 2.0]])!
	if _ := nanmax_axis(values, 2) {
		assert false, 'axis reduction must reject out-of-range axes'
	}
}

fn test_nan_multi_axis_reductions_and_shapes() ! {
	values := vtl.from_array([math.nan(), 2.0, 3.0, 4.0, 5.0, math.nan(), 7.0, 8.0], [
		2,
		2,
		2,
	])!
	assert nansum_axes(values, [0, -1], false)!.shape == [2]
	assert nansum_axes(values, [0, -1], false)!.to_array() == [7.0, 22.0]
	assert nanprod_axes(values, [0, 2], true)!.shape == [1, 2, 1]
	assert nanprod_axes(values, [0, 2], true)!.to_array() == [10.0, 672.0]
	assert nanmin_axes(values, [0, 2], false)!.to_array() == [2.0, 3.0]
	assert nanmax_axes(values, [0, 2], false)!.to_array() == [5.0, 8.0]
	assert nanmax_axes(values, [], false)!.to_array()[1] == 2.0
	if _ := nanmax_axes(values, [0, -3], false) {
		assert false, 'duplicate axes must return an error'
	}
	empty_axis := vtl.empty[f64]([2, 0])
	assert nansum_axes(empty_axis, [1], false)!.to_array() == [0.0, 0.0]
	assert nanprod_axes(empty_axis, [1], false)!.to_array() == [1.0, 1.0]
	assert math.is_nan(nanmin_axes(empty_axis, [1], false)!.get([0]))
	assert math.is_nan(nanmax_axes(empty_axis, [1], false)!.get([1]))
}

fn test_nan_sum_product_explicit_accumulator_types() {
	values := vtl.from_1d([f32(1e8), f32(1.0), f32(math.nan()), f32(-1e8)])!
	assert nansum_as[f32, f64](values) == 1.0
	products := vtl.from_1d([f32(1e20), f32(1e-20), f32(math.nan())])!
	assert math.abs(nanprod_as[f32, f64](products) - 1.0) < 1e-6
	assert nansum_as[f32, f64](vtl.from_1d([]f32{})!) == 0.0
	assert nanprod_as[f32, f64](vtl.from_1d([f32(math.nan())])!) == 1.0
}

fn test_nan_sum_product_explicit_accumulator_types_for_axis_reductions() ! {
	values := vtl.from_array([f32(1e8), f32(math.nan()), f32(1), f32(-1e8), f32(2), f32(math.nan()),
		f32(3), f32(4)], [2, 2, 2])!
	sums := nansum_axis_as[f32, f64](values, -1, false)!
	assert sums.shape == [2, 2]
	assert sums.to_array() == [1e8, -99999999.0, 2.0, 7.0]
	kept := nansum_axis_as[f32, f64](values, 1, true)!
	assert kept.shape == [2, 1, 2]
	assert kept.to_array() == [100000001.0, -1e8, 5.0, 4.0]

	products := nanprod_axis_as[f32, f64](values, 1, false)!
	assert products.to_array() == [1e8, -1e8, 6.0, 4.0]
}

fn test_nan_sum_product_explicit_accumulator_types_for_multi_axis_reductions() ! {
	values := vtl.from_array([f32(1e8), f32(math.nan()), f32(1), f32(-1e8), f32(2), f32(math.nan()),
		f32(3), f32(4)], [2, 2, 2])!
	sums := nansum_axes_as[f32, f64](values, [0, -1], false)!
	assert sums.shape == [2]
	assert sums.to_array() == [100000002.0, -99999992.0]
	kept := nansum_axes_as[f32, f64](values, [0, -1], true)!
	assert kept.shape == [1, 2, 1]
	assert kept.to_array() == sums.to_array()
	products := nanprod_axes_as[f32, f64](values, [0, 2], true)!
	assert products.shape == [1, 2, 1]
	assert products.to_array() == [200000000.0, -1200000000.0]

	unchanged := nansum_axes_as[f32, f64](values, [], false)!
	assert unchanged.shape == values.shape
	assert unchanged.get_nth(2) == 1.0
	assert math.is_nan(unchanged.get_nth(1))
	if _ := nansum_axes_as[f32, f64](values, [0, -3], false) {
		assert false, 'dtype-controlled NaN reduction must reject duplicate axes'
	}
}

fn test_nan_extrema_explicit_output_preserves_integer_precision() ! {
	values := vtl.from_array([
		i64(9_007_199_254_740_993),
		i64(9_007_199_254_740_992),
		i64(9_007_199_254_740_992),
		i64(9_007_199_254_740_993),
	], [2, 2])!

	assert nanmin_as[i64, i64](values)! == i64(9_007_199_254_740_992)
	assert nanmax_as[i64, i64](values)! == i64(9_007_199_254_740_993)
	assert nanmin_axis_as[i64, i64](values, 1, false)!.to_array() == [
		i64(9_007_199_254_740_992),
		i64(9_007_199_254_740_992),
	]
	assert nanmax_axis_as[i64, i64](values, 1, true)!.to_array() == [
		i64(9_007_199_254_740_993),
		i64(9_007_199_254_740_993),
	]
	assert nanmin_axes_as[i64, i64](values, [0, 1], true)!.get_nth(0) == i64(9_007_199_254_740_992)
	assert nanmax_axes_as[i64, i64](values, [0, 1], false)!.get_nth(0) == i64(9_007_199_254_740_993)

	boolean_values := vtl.from_1d([true, false])!
	assert !nanmin_as[bool, bool](boolean_values)!
	assert nanmax_as[bool, bool](boolean_values)!
}

fn test_nan_extrema_explicit_output_handles_missing_values_and_empty_axes() ! {
	values := vtl.from_array([
		f32(math.nan()),
		f32(2.0),
		f32(math.nan()),
		f32(math.nan()),
	], [2, 2])!
	minimum := nanmin_axis_as[f32, f32](values, 1, true)!
	assert minimum.shape == [2, 1]
	assert minimum.get([0, 0]) == 2.0
	assert math.is_nan(minimum.get([1, 0]))

	maximum := nanmax_axes_as[f32, f64](values, [1], false)!
	assert maximum.shape == [2]
	assert maximum.get([0]) == 2.0
	assert math.is_nan(maximum.get([1]))

	unchanged := nanmin_axes_as[f32, f32](values, [], false)!
	assert unchanged.shape == values.shape
	assert unchanged.get_nth(1) == 2.0
	assert math.is_nan(unchanged.get_nth(0))
	assert math.is_nan(nanmin_as[f32, f64](vtl.from_1d([f32(math.nan())])!)!)

	if _ := nanmin_as[f32, i64](vtl.from_1d([f32(math.nan())])!) {
		assert false, 'integer output cannot represent an empty NaN-aware extremum'
	}
	if _ := nanmax_axis_as[f32, i64](vtl.empty[f32]([1, 0]), 1, false) {
		assert false, 'integer output cannot represent an empty NaN-aware slice'
	}
}
