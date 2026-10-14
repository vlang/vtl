module core

import vtl

fn test_ravel_multi_index_matches_numpy_c_and_fortran_examples() ! {
	first := vtl.from_1d([3, 6, 6])!
	second := vtl.from_1d([4, 5, 1])!
	coordinates := [first, second]

	c_order := vtl.ravel_multi_index(coordinates, [7, 6])!
	f_order := vtl.ravel_multi_index(coordinates, [7, 6], order: .f)!
	assert c_order.shape == [3]
	assert c_order.to_array() == [22, 41, 37]
	assert f_order.to_array() == [31, 41, 13]
}

fn test_ravel_multi_index_broadcasts_coordinate_tensors() ! {
	first := vtl.from_array([0, 1], [2, 1])!
	second := vtl.from_array([1, 2, 0], [1, 3])!
	coordinates := [first, second]

	c_order := vtl.ravel_multi_index(coordinates, [2, 3])!
	f_order := vtl.ravel_multi_index(coordinates, [2, 3], order: .f)!
	assert c_order.shape == [2, 3]
	assert c_order.to_array() == [1, 2, 0, 4, 5, 3]
	assert f_order.to_array() == [2, 4, 0, 3, 5, 1]
}

fn test_ravel_multi_index_supports_global_and_per_axis_modes() ! {
	first := vtl.from_1d([-1, 4])!
	second := vtl.from_1d([6, -1])!
	coordinates := [first, second]
	assert vtl.ravel_multi_index(coordinates, [4, 6], mode: .wrap)!.to_array() == [18, 5]
	assert vtl.ravel_multi_index(coordinates, [4, 6], mode: .clip)!.to_array() == [5, 18]
	assert vtl.ravel_multi_index(coordinates, [4, 6], axis_modes: [.clip, .wrap])!.to_array() == [
		0,
		23,
	]
}

fn test_unravel_index_matches_c_and_fortran_order() ! {
	indices := vtl.from_1d([1, 5])!
	c_order := vtl.unravel_index(indices, [2, 3])!
	f_order := vtl.unravel_index(indices, [2, 3], order: .f)!
	assert c_order.len == 2
	assert c_order[0].shape == [2]
	assert c_order[0].to_array() == [0, 1]
	assert c_order[1].to_array() == [1, 2]
	assert f_order[0].to_array() == [1, 1]
	assert f_order[1].to_array() == [0, 2]
}

fn test_ravel_unravel_round_trip_with_broadcast_shapes_and_scalar_indices() ! {
	first := vtl.from_array([0, 1], [2, 1])!
	second := vtl.from_array([1, 2, 0], [1, 3])!
	coordinates := [first, second]
	flat := vtl.ravel_multi_index(coordinates, [2, 3])!
	unraveled := vtl.unravel_index(flat, [2, 3])!
	assert unraveled[0].shape == [2, 3]
	assert unraveled[0].to_array() == [0, 0, 0, 1, 1, 1]
	assert unraveled[1].to_array() == [1, 2, 0, 1, 2, 0]

	scalar_coordinate := vtl.tensor(1, [])
	scalar_coordinates := [scalar_coordinate, vtl.tensor(2, [])]
	scalar_flat := vtl.ravel_multi_index(scalar_coordinates, [2, 3])!
	assert scalar_flat.shape == []
	assert scalar_flat.get_nth[int](0) == 5
	scalar_unraveled := vtl.unravel_index(scalar_flat, [2, 3])!
	assert scalar_unraveled[0].shape == []
	assert scalar_unraveled[0].get_nth[int](0) == 1
	assert scalar_unraveled[1].get_nth[int](0) == 2
}

fn test_ravel_unravel_multi_index_handles_empty_coordinates() ! {
	empty := vtl.from_array([]int{}, [0])!
	coordinates := [empty, empty]
	traveled := vtl.ravel_multi_index(coordinates, [0, 3])!
	assert traveled.shape == [0]
	unraveled := vtl.unravel_index(traveled, [0, 3])!
	assert unraveled.len == 2
	assert unraveled[0].shape == [0]
	assert unraveled[1].shape == [0]
}

fn test_ravel_multi_index_rejects_invalid_shapes_modes_and_coordinates() {
	valid := vtl.from_1d([0, 1])!
	if _ := vtl.ravel_multi_index([valid], [2, 3]) {
		assert false, 'coordinate count mismatch must fail'
	}
	if _ := vtl.ravel_multi_index([valid, valid], [2, 3], axis_modes: [.clip]) {
		assert false, 'axis mode count mismatch must fail'
	}
	if _ := vtl.ravel_multi_index([valid, valid], [2, -3]) {
		assert false, 'negative dimension must fail'
	}
	if _ := vtl.ravel_multi_index([vtl.from_1d([0])!, vtl.from_1d([0])!], [max_int, 2]) {
		assert false, 'dimension product overflow must fail'
	}
	if _ := vtl.ravel_multi_index([vtl.from_1d([-1])!, vtl.from_1d([0])!], [2, 3]) {
		assert false, 'raise mode must reject negative coordinates'
	}
	if _ := vtl.ravel_multi_index([vtl.from_1d([0, 1])!, vtl.from_1d([0, 1, 2])!], [2, 3]) {
		assert false, 'incompatible coordinate shapes must fail'
	}
	if _ := vtl.ravel_multi_index([valid, vtl.from_1d([0, 3])!], [2, 3]) {
		assert false, 'raise mode must reject coordinates beyond an axis'
	}
}

fn test_unravel_index_rejects_invalid_dimensions_and_flat_indices() {
	indices := vtl.from_1d([-1, 6])!
	if _ := vtl.unravel_index(indices, [2, 3]) {
		assert false, 'negative and out-of-range flat indices must fail'
	}
	empty := vtl.from_array([]int{}, [0])!
	if _ := vtl.unravel_index(empty, [2, -3]) {
		assert false, 'negative dimensions must fail even for empty indices'
	}
	if _ := vtl.unravel_index(indices, [0, 3]) {
		assert false, 'non-empty indices cannot address an empty shape'
	}
	if _ := vtl.unravel_index(vtl.from_1d([0])!, [max_int, 2]) {
		assert false, 'dimension product overflow must fail'
	}
}
