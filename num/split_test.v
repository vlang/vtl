import vtl.num

fn test_array_split() {
	a := num.seq(9).reshape([3, 3])
	e1 := num.from_int([0, 1, 2], [1, 3])
	e2 := num.from_int([3, 4, 5], [1, 3])
	e3 := num.from_int([6, 7, 8], [1, 3])
	result := num.array_split(a, 3, 0)
	assert num.allclose(e1, result[0])
	assert num.allclose(e2, result[1])
	assert num.allclose(e3, result[2])
}

fn test_array_split_expl() {
	a := num.seq(9).reshape([3, 3])
	e1 := num.from_int([0, 3, 6], [3, 1])
	e2 := num.from_int([1, 2, 4, 5, 7, 8], [3, 2])
	result := num.array_split_expl(a, [1], 1)
	assert num.allclose(e1, result[0])
	assert num.allclose(e2, result[1])
}
