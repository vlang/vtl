module stats

import math
import vtl

fn test_explicit_accumulator_dtype_controls_global_sums_and_products() {
	values := vtl.from_1d([f32(1e8), 1.0, -1e8])!
	assert sum_as[f32, f64](values) == 1.0
	assert product_as[f32, f64](vtl.from_1d([f32(2), 3, 4])!) == 24.0
	assert sum_as[f32, f64](vtl.from_1d([]f32{})!) == 0.0
	assert product_as[f32, f64](vtl.from_1d([]f32{})!) == 1.0
}

fn test_explicit_accumulator_dtype_axis_reductions_and_shapes() ! {
	values := vtl.from_array([f32(1), 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	sums := sum_along_axes_as[f32, f64](values, [0, -1], false)!
	assert sums.shape == [2]
	assert sums.to_array() == [14.0, 22.0]
	products := product_along_axes_as[f32, f64](values, [-1, 0], true)!
	assert products.shape == [1, 2, 1]
	assert products.to_array() == [60.0, 672.0]
	assert sum_along_axes_as[f32, f64](values, [0, 1, 2], false)!.get_nth(0) == 36.0
	all_product := product_along_axes_as[f32, f64](values, [0, 1, 2], true)!
	assert all_product.shape == [1, 1, 1]
	assert all_product.get_nth(0) == 40320.0
	axis_sum := sum_along_axis_as[f32, f64](values, 1, true)!
	assert axis_sum.shape == [2, 1, 2]
	assert axis_sum.to_array() == [4.0, 6.0, 12.0, 14.0]
	axis_product := product_along_axis_as[f32, f64](values, -1, false)!
	assert axis_product.shape == [2, 2]
	assert axis_product.to_array() == [2.0, 12.0, 30.0, 56.0]
	unchanged := sum_along_axes_as[f32, f64](values, [], false)!
	assert unchanged.shape == values.shape
	assert unchanged.to_array() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
	empty := vtl.zeros[f32]([2, 0, 3], vtl.TensorData{})
	assert sum_along_axes_as[f32, f64](empty, [1], false)!.to_array() == [0.0, 0.0, 0.0, 0.0, 0.0,
		0.0]
	assert product_along_axes_as[f32, f64](empty, [1], false)!.to_array() == [1.0, 1.0, 1.0, 1.0,
		1.0, 1.0]
	if _ := sum_along_axes_as[f32, f64](values, [0, -3], false) {
		assert false, 'duplicate axes must return an error'
	}
	if _ := product_along_axis_as[f32, f64](values, 3, true) {
		assert false, 'out-of-range axes must return an error'
	}
	assert !math.is_nan(sum_as[f32, f64](values))
}
