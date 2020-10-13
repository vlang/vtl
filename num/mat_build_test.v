import vnum.num

fn test_diag() {
	a := num.from_int_1d([1, 2, 3])
	expected := num.from_int([1, 0, 0, 0, 2, 0, 0, 0, 3], [3, 3])
	result := num.diag(a)
	assert num.allclose(result, expected)
}

fn test_diag_flat() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	expected := num.from_int([1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4], [4, 4])
	result := num.diag_flat(a)
	assert num.allclose(result, expected)
}

fn test_tril() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	expected := num.from_int([1, 0, 3, 4], [2, 2])
	result := num.tril(a)
	assert num.allclose(result, expected)
}

fn test_triu() {
	a := num.from_int([1, 2, 3, 4], [2, 2])
	expected := num.from_int([1, 2, 0, 4], [2, 2])
	result := num.triu(a)
	assert num.allclose(result, expected)
}

fn test_tril_offset() {
	a := num.from_int([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])
	expected := num.from_int([0, 0, 0, 4, 0, 0, 7, 8, 0], [3, 3])
	result := num.tril_offset(a, -1)
	assert num.allclose(result, expected)
}

fn test_triu_offset() {
	a := num.from_int([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])
	expected := num.from_int([0, 2, 3, 0, 0, 6, 0, 0, 0], [3, 3])
	result := num.triu_offset(a, 1)
	assert num.allclose(result, expected)
}
