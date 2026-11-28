module interop

import vtl.interop as buffer

fn test_contiguous_view_metadata_and_values() ! {
	backing := [f64(1), 2, 3, 4]
	view := buffer.contiguous(backing, [2, 2])!
	assert view.strides == [2, 1]
	assert view.offset == 0
	assert view.mutable
	assert view.values() == [f64(1), 2, 3, 4]
	assert view.is_zero_copy_eligible()

	meta := view.metadata()
	assert meta.shape == [2, 2]
	assert meta.strides == [2, 1]
	assert meta.element_size == 8
	assert meta.mutable
}

fn test_borrow_supports_strided_and_offset_views() ! {
	backing := [f64(1), 2, 3, 4, 5, 6, 7, 8]
	view := buffer.borrow(backing, [2, 2], [4, 1], 0, false)!
	assert view.values() == [f64(1), 2, 5, 6]
	assert !view.mutable
	assert !view.is_zero_copy_eligible()

	offset_view := buffer.borrow(backing, [2], [2], 1, true)!
	assert offset_view.values() == [f64(2), 4]
	assert !offset_view.is_zero_copy_eligible()
}

fn test_borrow_rejects_bad_layouts() ! {
	buffer.borrow([f64(1), 2, 3], [2], [1], 3, true) or {
		assert err.msg().contains('outside the buffer')
		return
	}
	assert false, 'out-of-bounds view should be rejected'
}

fn test_borrow_rejects_overlapping_strides() ! {
	buffer.borrow([f64(1), 2, 3, 4], [2, 2], [1, 1], 0, true) or {
		assert err.msg().contains('overlapping stride')
		return
	}
	assert false, 'overlapping strides should be rejected'
}

fn test_borrow_rejects_invalid_arguments() ! {
	buffer.borrow([f64(1)], [1], [1, 1], 0, true) or {
		assert err.msg().contains('same rank')
		return
	}
	assert false, 'rank mismatch should be rejected'

	buffer.borrow([f64(1)], [1], [0], 0, true) or {
		assert err.msg().contains('positive')
		return
	}
	assert false, 'non-positive stride should be rejected'

	buffer.borrow([f64(1)], [1], [1], -1, true) or {
		assert err.msg().contains('non-negative')
		return
	}
	assert false, 'negative offset should be rejected'
}

fn test_copy_fallback_is_independent() ! {
	mut backing := [f64(1), 2, 3, 4]
	view := buffer.contiguous(backing, [2, 2])!
	tensor := view.to_tensor()!
	backing[0] = 99
	assert tensor.get_nth[f64](0) == 1
	assert tensor.to_array() == [f64(1), 2, 3, 4]
}

fn test_zero_copy_wrap_shares_buffer() ! {
	mut backing := [f64(1), 2, 3, 4]
	view := buffer.contiguous(backing, [2, 2])!
	tensor := view.as_borrowed_tensor()!
	backing[0] = 99
	assert tensor.get_nth[f64](0) == 99
	assert tensor.strides == [2, 1]
}

fn test_non_eligible_view_rejects_zero_copy() ! {
	backing := [f64(1), 2, 3, 4, 5, 6]
	view := buffer.borrow(backing, [2, 2], [3, 1], 0, true)!
	view.as_borrowed_tensor() or {
		assert err.msg().contains('use to_tensor instead')
		return
	}
	assert false, 'strided view should require a copy'
}

fn test_empty_shape_has_zero_elements() ! {
	view := buffer.contiguous([]f64{}, [0, 3])!
	assert buffer.element_count([0, 3]) == 0
	assert view.values().len == 0
	assert view.is_zero_copy_eligible()
}
