module vtl

fn test_diag() {
	t := from_1d([1, 2, 3, 4, 5, 6, 7, 8, 9])!
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

fn test_diag_flat_from_1d() {
	t := from_1d([1, 2, 3, 4, 5, 6, 7, 8, 9])!
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
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
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
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
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
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	tril := t.tril_offset(0)
	assert tril.array_equal(t.tril())
}

fn test_tril_offset_1() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
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
	mut t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
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
	mut t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	t.tril_inplace_offset(0)
	assert t.array_equal(from_2d([[1, 0, 0], [4, 5, 0], [7, 8, 9]])!)
}

fn test_tril_inplace_offset_1() {
	mut t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	t.tril_inplace_offset(1)
	assert t.array_equal(from_2d([[1, 2, 0], [4, 5, 6], [7, 8, 9]])!)
}

fn test_triu() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
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
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	triu := t.triu_offset(0)
	assert triu.array_equal(t.triu())
}

fn test_triu_offset_1() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
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
	mut t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
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

fn test_triu_inplace_offset_0() {
	mut t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	t.triu_inplace_offset(0)
	assert t.array_equal(from_2d([[1, 2, 3], [0, 5, 6], [0, 0, 9]])!)
}

fn test_triu_inplace_offset_1() {
	mut t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.shape == [3, 3]
	assert t.strides == [3, 1]
	t.triu_inplace_offset(1)
	assert t.array_equal(from_2d([[0, 2, 3], [0, 0, 6], [0, 0, 0]])!)
}
