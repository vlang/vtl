module core

import vtl

fn test_one_hot_encodes_vector_and_preserves_trailing_class_axis() ! {
	labels := vtl.from_1d([2, 0, 1, 2])!
	encoded := vtl.one_hot(labels, 3)!
	assert encoded.shape == [4, 3]
	assert encoded.to_array() == [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1]
}

fn test_one_hot_encodes_arbitrary_rank_and_scalar_indices() ! {
	labels := vtl.from_2d([[0, 2], [1, 0]])!
	encoded := vtl.one_hot(labels, 3)!
	assert encoded.shape == [2, 2, 3]
	assert encoded.to_array() == [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]

	scalar := vtl.from_array([2], [])!
	assert vtl.one_hot(scalar, 4)!.shape == [4]
	assert vtl.one_hot(scalar, 4)!.to_array() == [0, 0, 1, 0]
}

fn test_one_hot_accepts_empty_indices() ! {
	labels := vtl.zeros[int]([0])
	encoded := vtl.one_hot(labels, 5)!
	assert encoded.shape == [0, 5]
	assert encoded.size == 0
}

fn test_one_hot_rejects_invalid_class_count_and_indices() ! {
	labels := vtl.from_1d([0, 3])!
	for num_classes in [0, -1] {
		vtl.one_hot(labels, num_classes) or {
			assert err.msg().contains('num_classes must be positive')
			continue
		}
		assert false, 'non-positive class counts must fail'
	}
	assert_one_hot_rejects_class(vtl.from_1d([-1])!)
	assert_one_hot_rejects_class(vtl.from_1d([3])!)
}

fn assert_one_hot_rejects_class(labels &vtl.Tensor[int]) {
	vtl.one_hot(labels, 3) or {
		assert err.msg().contains('outside [0, 3)')
		return
	}
	assert false, 'out-of-range class indices must fail'
}
