module core

import vtl

fn test_indices_builds_numpy_ordered_coordinate_planes() ! {
	coordinates := vtl.indices([2, 3])!
	assert coordinates.shape == [2, 2, 3]
	assert coordinates.to_array() == [0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 2]

	line := vtl.indices([4])!
	assert line.shape == [1, 4]
	assert line.to_array() == [0, 1, 2, 3]
}

fn test_indices_handles_zero_rank_and_empty_dimensions() ! {
	scalar_shape := vtl.indices([])!
	assert scalar_shape.shape == [0]
	assert scalar_shape.size == 0

	empty := vtl.indices([2, 0, 3])!
	assert empty.shape == [3, 2, 0, 3]
	assert empty.size == 0
}

fn test_indices_creation_options_preserve_column_major_layout() ! {
	dense := vtl.indices([2, 3], memory: .col_major)!
	assert dense.memory == .col_major
	assert dense.strides == [1, 2, 4]
	assert dense.to_array() == [0, 0, 0, 1, 1, 1, 0, 1, 2, 0, 1, 2]

	sparse := vtl.indices_sparse([2, 3], memory: .col_major)!
	assert sparse[0].memory == .col_major
	assert sparse[1].memory == .col_major
	assert sparse[0].to_array() == [0, 1]
	assert sparse[1].to_array() == [0, 1, 2]
}

fn test_indices_rejects_negative_and_overflowing_dimensions() {
	if _ := vtl.indices([2, -1]) {
		assert false, 'indices must reject negative dimensions'
	}
	if _ := vtl.indices([max_int, 2]) {
		assert false, 'indices must reject overflowing shapes'
	}
}
