module core

import vtl

fn test_pad_constant_and_per_axis_widths() ! {
	values := vtl.from_array([1, 2, 3, 4], [2, 2])!
	result := vtl.pad[int](values, [[1, 0], [0, 1]], .constant, 0)!
	assert result.shape == [3, 3]
	assert result.to_array() == [0, 0, 0, 1, 2, 0, 3, 4, 0]
}

fn test_pad_boundary_modes() ! {
	values := vtl.from_1d([1, 2, 3])!
	assert vtl.pad[int](values, [[2, 2]], .edge, 0)!.to_array() == [1, 1, 1, 2, 3, 3, 3]
	assert vtl.pad[int](values, [[2, 2]], .wrap, 0)!.to_array() == [2, 3, 1, 2, 3, 1, 2]
	assert vtl.pad[int](values, [[2, 2]], .reflect, 0)!.to_array() == [3, 2, 1, 2, 3, 2, 1]
	assert vtl.pad[int](values, [[2, 2]], .symmetric, 0)!.to_array() == [2, 1, 1, 2, 3, 3, 2]
	one := vtl.from_1d([7])!
	assert vtl.pad[int](one, [[3, 2]], .reflect, 0)!.to_array() == [7, 7, 7, 7, 7, 7]
}

fn test_pad_empty_input_and_validation() ! {
	empty := vtl.empty[int]([0])
	assert vtl.pad[int](empty, [[2, 1]], .constant, 9)!.to_array() == [9, 9, 9]
	if _ := vtl.pad[int](empty, [[1, 1]], .edge, 0) {
		assert false, 'edge padding must reject empty input dimensions'
	}
	values := vtl.from_1d([1, 2])!
	if _ := vtl.pad[int](values, [], .constant, 0) {
		assert false, 'padding must provide one width pair per dimension'
	}
	if _ := vtl.pad[int](values, [[1]], .constant, 0) {
		assert false, 'padding width pairs must contain two values'
	}
	if _ := vtl.pad[int](values, [[-1, 0]], .constant, 0) {
		assert false, 'padding widths must be non-negative'
	}
	if _ := vtl.pad[int](values, [[max_int, 0]], .constant, 0) {
		assert false, 'padded dimensions must not overflow int'
	}
}
