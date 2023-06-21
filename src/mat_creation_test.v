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
