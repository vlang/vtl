module core

import vtl

fn test_diag() {
	t := vtl.from_1d([1, 2, 3, 4, 5, 6, 7, 8, 9])!
	assert t.shape == [9]
	assert t.strides == [1]
	diag := t.diag()!
	assert diag.shape == [9]
	assert diag.strides == [10]
	assert diag.get([0]) == 1
	assert diag.get([1]) == 2
	assert diag.get([2]) == 3
	assert diag.get([3]) == 4
	assert diag.get([4]) == 5
	assert diag.get([5]) == 6
	assert diag.get([6]) == 7
	assert diag.get([7]) == 8
	assert diag.get([8]) == 9
}

fn test_tril_and_triu_support_batched_matrices_and_offsets() ! {
	input := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3])!
	lower := vtl.tril(input, k: 0)!
	upper := vtl.triu(input, k: 1)!
	assert lower.shape == input.shape
	assert vtl.tril(input)!.array_equal(lower)
	assert vtl.triu(input)!.array_equal(vtl.from_array([1, 2, 3, 0, 5, 6, 7, 8, 9, 0, 11, 12],
		[2, 2, 3])!)
	assert lower.array_equal(vtl.from_array([1, 0, 0, 4, 5, 0, 7, 0, 0, 10, 11, 0], [
		2,
		2,
		3,
	])!)
	assert upper.array_equal(vtl.from_array([0, 2, 3, 0, 0, 6, 0, 8, 9, 0, 0, 12], [2, 2, 3])!)
	assert input.get_nth(1) == 2
	lower.set_nth(0, 99)
	assert input.get_nth(0) == 1
}

fn test_tril_and_triu_reject_rank_below_two() {
	vector := vtl.from_1d([1, 2, 3])!
	if _ := vtl.tril(vector, k: 0) {
		assert false, 'tril must reject rank-one input'
	}
	if _ := vtl.triu(vector, k: 0) {
		assert false, 'triu must reject rank-one input'
	}
}

fn test_tril_supports_boolean_tensors() ! {
	input := vtl.from_2d([[true, true], [true, false]])!
	upper := vtl.triu(input, k: 1)!
	assert upper.array_equal(vtl.from_2d([[false, true], [false, false]])!)
}

fn test_diag_flat_from_1d() {
	t := vtl.from_1d([1, 2, 3, 4, 5, 6, 7, 8, 9])!
	assert t.shape == [9]
	assert t.strides == [1]
	diag := t.diag_flat()!
	assert diag.shape == [9]
	assert diag.strides == [10]
	assert diag.get([0]) == 1
	assert diag.get([1]) == 2
	assert diag.get([2]) == 3
	assert diag.get([3]) == 4
	assert diag.get([4]) == 5
	assert diag.get([5]) == 6
	assert diag.get([6]) == 7
	assert diag.get([7]) == 8
	assert diag.get([8]) == 9
}

fn test_diag_flat_from_2d() {
	t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	diag := t.diag_flat()!
	assert diag.shape == [9]
	assert diag.strides == [10]
	assert diag.get([0]) == 1
	assert diag.get([1]) == 2
	assert diag.get([2]) == 3
	assert diag.get([3]) == 4
	assert diag.get([4]) == 5
	assert diag.get([5]) == 6
	assert diag.get([6]) == 7
	assert diag.get([7]) == 8
	assert diag.get([8]) == 9
}

fn test_tril() {
	t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	tril := t.tril()
	assert tril.shape == [3, 3]
	assert tril.strides == [3, 1]
	assert tril.get([0, 0]) == 1
	assert tril.get([0, 1]) == 0
	assert tril.get([0, 2]) == 0
	assert tril.get([1, 0]) == 4
	assert tril.get([1, 1]) == 5
	assert tril.get([1, 2]) == 0
	assert tril.get([2, 0]) == 7
	assert tril.get([2, 1]) == 8
	assert tril.get([2, 2]) == 9
}

fn test_tril_offset_0() {
	t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	tril := t.tril_offset(0)
	assert tril.array_equal(t.tril())
}

fn test_tril_offset_1() {
	t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	tril := t.tril_offset(1)
	assert tril.shape == [3, 3]
	assert tril.strides == [3, 1]
	assert tril.get([0, 0]) == 1
	assert tril.get([0, 1]) == 2
	assert tril.get([0, 2]) == 0
	assert tril.get([1, 0]) == 4
	assert tril.get([1, 1]) == 5
	assert tril.get([1, 2]) == 6
	assert tril.get([2, 0]) == 7
	assert tril.get([2, 1]) == 8
	assert tril.get([2, 2]) == 9
}

fn test_tril_inplace() {
	mut t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	t.tril_inplace()
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	assert t.get([0, 0]) == 1
	assert t.get([0, 1]) == 0
	assert t.get([0, 2]) == 0
	assert t.get([1, 0]) == 4
	assert t.get([1, 1]) == 5
	assert t.get([1, 2]) == 0
	assert t.get([2, 0]) == 7
	assert t.get([2, 1]) == 8
	assert t.get([2, 2]) == 9
}

fn test_tril_inplace_offset_0() {
	mut t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	t.tril_inpl_offset(0)
	assert t.array_equal(vtl.from_2d([[1, 0, 0], [4, 5, 0], [7, 8, 9]])!)
}

fn test_tril_inplace_offset_1() {
	mut t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	t.tril_inpl_offset(1)
	assert t.array_equal(vtl.from_2d([[1, 2, 0], [4, 5, 6], [7, 8, 9]])!)
}

fn test_triu() {
	t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	triu := t.triu()
	assert triu.shape == [3, 3]
	assert triu.strides == [3, 1]
	assert triu.get([0, 0]) == 1
	assert triu.get([0, 1]) == 2
	assert triu.get([0, 2]) == 3
	assert triu.get([1, 0]) == 0
	assert triu.get([1, 1]) == 5
	assert triu.get([1, 2]) == 6
	assert triu.get([2, 0]) == 0
	assert triu.get([2, 1]) == 0
	assert triu.get([2, 2]) == 9
}

fn test_triu_offset_0() {
	t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	triu := t.triu_offset(0)
	assert triu.array_equal(t.triu())
}

fn test_triu_offset_1() {
	t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	triu := t.triu_offset(1)
	assert triu.shape == [3, 3]
	assert triu.strides == [3, 1]
	assert triu.get([0, 0]) == 0
	assert triu.get([0, 1]) == 2
	assert triu.get([0, 2]) == 3
	assert triu.get([1, 0]) == 0
	assert triu.get([1, 1]) == 0
	assert triu.get([1, 2]) == 6
	assert triu.get([2, 0]) == 0
	assert triu.get([2, 1]) == 0
	assert triu.get([2, 2]) == 0
}

fn test_triu_inplace() {
	mut t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	t.triu_inplace()
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	assert t.get([0, 0]) == 1
	assert t.get([0, 1]) == 2
	assert t.get([0, 2]) == 3
	assert t.get([1, 0]) == 0
	assert t.get([1, 1]) == 5
	assert t.get([1, 2]) == 6
	assert t.get([2, 0]) == 0
	assert t.get([2, 1]) == 0
	assert t.get([2, 2]) == 9
}

fn test_triu_offset_0_matches_inplace() {
	t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	triu := t.triu_offset(0)
	assert triu.array_equal(vtl.from_2d([[1, 2, 3], [0, 5, 6], [0, 0, 9]])!)
}

fn test_triu_offset_1_matches_inplace() {
	t := vtl.from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	triu := t.triu_offset(1)
	assert triu.array_equal(vtl.from_2d([[0, 2, 3], [0, 0, 6], [0, 0, 0]])!)
}
