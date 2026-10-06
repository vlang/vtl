module stats

import vtl

fn test_gradient_axis_uses_central_and_one_sided_differences() ! {
	values := vtl.from_1d([0.0, 1.0, 4.0, 9.0])!
	gradient := gradient_axis[f64](values, 1.0, 0)!
	expected := vtl.from_1d([1.0, 2.0, 4.0, 5.0])!
	assert gradient.array_equal(expected)
}

fn test_gradient_axis_supports_negative_axis_and_spacing() ! {
	values := vtl.from_2d([[0.0, 1.0, 4.0], [0.0, 2.0, 8.0]])!
	gradient := gradient_axis[f64](values, 2.0, -1)!
	expected := vtl.from_2d([[0.5, 1.0, 1.5], [1.0, 2.0, 3.0]])!
	assert gradient.array_equal(expected)
}

fn test_gradient_axis_reads_noncontiguous_views() ! {
	values := vtl.from_2d([[0.0, 0.0], [1.0, 2.0], [4.0, 8.0]])!
	transposed := values.transpose([1, 0])!
	gradient := gradient_axis[f64](transposed, 1.0, 1)!
	expected := vtl.from_2d([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])!
	assert gradient.array_equal(expected)
}

fn test_gradient_axis_rejects_invalid_input() {
	scalar := vtl.from_array([3.0], [])!
	if _ := gradient_axis[f64](scalar, 1.0, 0) {
		assert false
	}
	values := vtl.from_1d([1.0, 2.0])!
	if _ := gradient_axis[f64](values, 0.0, 0) {
		assert false
	}
	if _ := gradient_axis[f64](values, 1.0, 1) {
		assert false
	}
	short := vtl.from_1d([1.0])!
	if _ := gradient_axis[f64](short, 1.0, 0) {
		assert false
	}
}
