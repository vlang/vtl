module main

import vtl

fn test_take_axis() {
	t := vtl.from_array([0, 1, 2, 3, 4, 5], [2, 3])!
	taken := t.take([2, 0, -1], 1)!
	assert taken.shape == [2, 3]
	assert taken.to_array() == [2, 0, 2, 5, 3, 5]

	rows := t.take([-1, 0], 0)!
	assert rows.shape == [2, 3]
	assert rows.to_array() == [3, 4, 5, 0, 1, 2]

	column_major := vtl.from_array([0, 3, 1, 4, 2, 5], [2, 3], memory: .col_major)!
	column_major_taken := column_major.take([2, 0], 1)!
	assert column_major_taken.is_col_major()
	assert column_major_taken.to_array() == [2, 0, 5, 3]

	columns := t.take([], -1)!
	assert columns.shape == [2, 0]
	assert columns.size() == 0
}

fn test_take_rejects_invalid_axes_and_indices() {
	t := vtl.ones[int]([2, 3])
	if _ := t.take([0], 2) {
		assert false, 'take must reject out-of-range axes'
	} else {
		assert true
	}
	if _ := t.take([3], 1) {
		assert false, 'take must reject out-of-range indices'
	} else {
		assert true
	}
	if _ := t.take([-3], 0) {
		assert false, 'take must reject negative indices below -axis_size'
	} else {
		assert true
	}
}

fn test_take_along_axis() {
	t := vtl.from_array([0, 1, 2, 3, 4, 5], [2, 3])!
	indices := vtl.from_array([2, 0, 1, -1], [2, 2])!
	taken := t.take_along_axis(indices, 1)!
	assert taken.shape == [2, 2]
	assert taken.to_array() == [2, 0, 4, 5]

	if _ := t.take_along_axis(vtl.ones[int]([1, 2]), 1) {
		assert false, 'take_along_axis must reject mismatched non-axis dimensions'
	} else {
		assert true
	}
}

fn test_get() {
	t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.get([0, 0]) == 1
	assert t.get([0, 1]) == 2
	assert t.get([1, 0]) == 3
	assert t.get([1, 1]) == 4
	assert t.get([2, 0]) == 5
	assert t.get([2, 1]) == 6
	assert t.get([3, 0]) == 7
	assert t.get([3, 1]) == 8
	assert t.get([4, 0]) == 9
	assert t.get([4, 1]) == 10
}

fn test_get_nth() {
	t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.get_nth(0) == 1
	assert t.get_nth(1) == 2
	assert t.get_nth(2) == 3
	assert t.get_nth(3) == 4
	assert t.get_nth(4) == 5
	assert t.get_nth(5) == 6
	assert t.get_nth(6) == 7
	assert t.get_nth(7) == 8
	assert t.get_nth(8) == 9
	assert t.get_nth(9) == 10
}

fn test_offset_index() {
	t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.offset_index([0, 0]) == 0
	assert t.offset_index([0, 1]) == 1
	assert t.offset_index([1, 0]) == 2
	assert t.offset_index([1, 1]) == 3
	assert t.offset_index([2, 0]) == 4
	assert t.offset_index([2, 1]) == 5
	assert t.offset_index([3, 0]) == 6
	assert t.offset_index([3, 1]) == 7
	assert t.offset_index([4, 0]) == 8
	assert t.offset_index([4, 1]) == 9
}

fn test_nth_index() {
	t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.nth_index(0) == [0, 0]
	assert t.nth_index(1) == [0, 1]
	assert t.nth_index(2) == [1, 0]
	assert t.nth_index(3) == [1, 1]
	assert t.nth_index(4) == [2, 0]
	assert t.nth_index(5) == [2, 1]
	assert t.nth_index(6) == [3, 0]
	assert t.nth_index(7) == [3, 1]
	assert t.nth_index(8) == [4, 0]
	assert t.nth_index(9) == [4, 1]
}
