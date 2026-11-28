module core

import vtl

fn test_clone_produces_independent_storage() ! {
	original := vtl.from_array([f64(1), 2, 3, 4], [2, 2])!
	copied := original.clone()
	assert copied.shape == original.shape
	assert copied.strides == original.strides
	assert copied.to_array() == original.to_array()

	copied.set_nth(0, 99)
	assert original.get_nth[f64](0) == 1
	assert copied.get_nth[f64](0) == 99
}

fn test_view_shares_storage_while_clone_does_not() ! {
	original := vtl.from_array([f64(1), 2, 3, 4], [2, 2])!
	aliased := original.view()
	aliased.set_nth(0, 42)
	assert original.get_nth[f64](0) == 42

	copied := original.clone()
	copied.set_nth(0, 7)
	assert original.get_nth[f64](0) == 42
}

fn test_clone_preserves_non_contiguous_strides_and_values() ! {
	original := vtl.from_array([f64(1), 2, 3, 4, 5, 6], [2, 3])!
	transposed := original.transpose([1, 0])!
	assert !transposed.is_row_major_contiguous()

	copied := transposed.clone()
	assert copied.shape == [3, 2]
	// Clone is owning and contiguous: strides are recalculated instead of
	// preserving the non-contiguous view layout.
	assert copied.is_row_major_contiguous()
	assert copied.strides == [2, 1]
	assert copied.to_array() == transposed.to_array()

	original.set_nth(0, 100)
	assert transposed.get_nth[f64](0) == 100
	assert copied.get_nth[f64](0) == 1
}

fn test_clone_of_sliced_view_copies_logical_values_contiguously() ! {
	original := vtl.from_array([f64(1), 2, 3, 4, 5, 6], [2, 3])!
	sliced := original.slice([0, 2], [0, 2])!
	assert sliced.shape == [2, 2]
	assert sliced.strides == [3, 1]
	assert !sliced.is_row_major_contiguous()

	copied := sliced.clone()
	assert copied.shape == [2, 2]
	assert copied.strides == [2, 1]
	assert copied.is_row_major_contiguous()
	assert copied.to_array() == [f64(1), 2, 4, 5]

	copied.set_nth(0, 99)
	assert sliced.get_nth[f64](0) == 1
	assert original.get_nth[f64](0) == 1
}

fn test_clone_preserves_column_major_layout() ! {
	original := vtl.from_array([f64(1), 2, 3, 4], [2, 2], memory: .col_major)!
	copied := original.clone()
	assert copied.memory == .col_major
	assert copied.strides == original.strides
	assert copied.to_array() == original.to_array()
}

fn test_clone_of_empty_tensor_is_independent() ! {
	original := vtl.from_array([]f64{}, [0, 3])!
	copied := original.clone()
	assert copied.size == 0
	assert copied.shape == [0, 3]
	assert copied.to_array().len == 0
}
