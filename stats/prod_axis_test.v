module stats

import vtl

fn test_prod_axis_uses_multiplicative_identity() ! {
	values := vtl.from_1d([2, 3, 4])!
	assert prod_axis[int](values, AxisData{ axis: 0 }) == 24
	assert prod_axis_with_dims[int](values, AxisData{ axis: 0 }) == 24
}

fn test_prod_axis_multiplies_values_on_selected_matrix_axis() ! {
	values := vtl.from_2d([[2, 3], [4, 5]])!
	assert prod_axis[int](values, AxisData{ axis: 0 }) == 8
	assert prod_axis[int](values, AxisData{ axis: 1 }) == 6
	assert prod_axis_with_dims[int](values, AxisData{ axis: 0 }) == 8
	assert prod_axis_with_dims[int](values, AxisData{ axis: 1 }) == 6
}

fn test_empty_prod_uses_multiplicative_identity() ! {
	empty := vtl.from_1d([]int{})!
	assert prod[int](empty) == 1
	assert prod_axis[int](empty, AxisData{ axis: 0 }) == 1
	assert prod_axis_with_dims[int](empty, AxisData{ axis: 0 }) == 1
}
