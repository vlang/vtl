module stats

import vtl

fn test_trapezoid_integrates_uniformly_spaced_values() ! {
	values := vtl.from_1d([1, 2, 3])!
	area := trapezoid[int](values, 1.0)!
	assert area.shape == []
	assert area.get_nth(0) == 4.0

	double_spacing := trapezoid[int](values, 2.0)!
	assert double_spacing.get_nth(0) == 8.0
}

fn test_trapezoid_reduces_selected_axis_and_respects_x_order() ! {
	values := vtl.from_2d([[0, 1, 2], [3, 4, 5]])!
	last_axis := trapezoid_axis[int](values, 1.0, -1)!
	assert last_axis.shape == [2]
	assert last_axis.to_array() == [2.0, 8.0]

	first_axis := trapezoid_axis[int](values, 1.0, 0)!
	assert first_axis.shape == [3]
	assert first_axis.to_array() == [1.5, 2.5, 3.5]

	y := vtl.from_1d([1.0, 2.0, 3.0])!
	x := vtl.from_1d([4, 6, 8])!
	assert trapezoid_x_axis[f64, int](y, x, 0)!.get_nth(0) == 8.0
	decreasing_x := vtl.from_1d([8, 6, 4])!
	assert trapezoid_x_axis[f64, int](y, decreasing_x, 0)!.get_nth(0) == -8.0
}

fn test_trapezoid_empty_and_single_sample_integrate_to_zero() ! {
	empty := trapezoid[int](vtl.from_1d([]int{})!, 1.0)!
	assert empty.shape == []
	assert empty.get_nth(0) == 0.0
	single := trapezoid[int](vtl.from_1d([7])!, 1.0)!
	assert single.get_nth(0) == 0.0
}

fn test_trapezoid_rejects_invalid_shapes_and_axes() ! {
	values := vtl.from_2d([[1, 2], [3, 4]])!
	if _ := trapezoid_x_axis[int, int](values, vtl.from_1d([0])!, 0) {
		assert false, 'trapezoid must reject x length mismatches'
	}
	if _ := trapezoid_axis[int](values, 1.0, 2) {
		assert false, 'trapezoid must reject out-of-range axes'
	}
	if _ := trapezoid[int](vtl.from_array([1], [])!, 1.0) {
		assert false, 'trapezoid must reject scalar inputs'
	}
}
