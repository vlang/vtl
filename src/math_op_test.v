module vtl

fn test_add() {
	a := from_1d([1, 2, 3, 4])!
	b := from_1d([0, 1, 2, 3])!
	result := a.add(b)!
	expected := from_1d([1, 3, 5, 7])!
	assert result.array_equal(expected)
}

fn test_add_scalar() {
	a := from_1d([1, 2, 3, 4])!
	result := a.add_scalar(1)!
	expected := from_1d([2, 3, 4, 5])!
	assert result.array_equal(expected)
}

fn test_add_2d() {
	a := from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	b := from_2d([[0, 1, 2, 3], [4, 5, 6, 7]])!
	result := a.add(b)!
	expected := from_2d([[1, 3, 5, 7], [9, 11, 13, 15]])!
	assert result.array_equal(expected)
}

fn test_add_2d_scalar() {
	a := from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	result := a.add_scalar(1)!
	expected := from_2d([[2, 3, 4, 5], [6, 7, 8, 9]])!
	assert result.array_equal(expected)
}

fn test_subtract() {
	a := from_1d([1, 2, 3, 4])!
	b := from_1d([0, 1, 2, 3])!
	result := a.subtract(b)!
	expected := from_1d([1, 1, 1, 1])!
	assert result.array_equal(expected)
}

fn test_subtract_scalar() {
	a := from_1d([1, 2, 3, 4])!
	result := a.subtract_scalar(1)!
	expected := from_1d([0, 1, 2, 3])!
	assert result.array_equal(expected)
}

fn test_subtract_2d() {
	a := from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	b := from_2d([[0, 1, 2, 3], [4, 5, 6, 7]])!
	result := a.subtract(b)!
	expected := from_2d([[1, 1, 1, 1], [1, 1, 1, 1]])!
	assert result.array_equal(expected)
}

fn test_subtract_2d_scalar() {
	a := from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	result := a.subtract_scalar(1)!
	expected := from_2d([[0, 1, 2, 3], [4, 5, 6, 7]])!
	assert result.array_equal(expected)
}

fn test_multiply() {
	a := from_1d([1, 2, 3, 4])!
	b := from_1d([0, 1, 2, 3])!
	result := a.multiply(b)!
	expected := from_1d([0, 2, 6, 12])!
	assert result.array_equal(expected)
}

fn test_multiply_scalar() {
	a := from_1d([1, 2, 3, 4])!
	result := a.multiply_scalar(2)!
	expected := from_1d([2, 4, 6, 8])!
	assert result.array_equal(expected)
}

fn test_multiply_2d() {
	a := from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	b := from_2d([[0, 1, 2, 3], [4, 5, 6, 7]])!
	result := a.multiply(b)!
	expected := from_2d([[0, 2, 6, 12], [20, 30, 42, 56]])!
	assert result.array_equal(expected)
}

fn test_multiply_2d_scalar() {
	a := from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	result := a.multiply_scalar(2)!
	expected := from_2d([[2, 4, 6, 8], [10, 12, 14, 16]])!
	assert result.array_equal(expected)
}

fn test_divide() {
	a := from_1d([1, 2, 3, 4])!
	b := from_1d([1, 2, 3, 4])!
	result := a.divide(b)!
	expected := from_1d([1, 1, 1, 1])!
	assert result.array_equal(expected)
}

fn test_divide_scalar() {
	a := from_1d([1.0, 2, 3, 4])!
	result := a.divide_scalar(2.0)!
	expected := from_1d([0.5, 1, 1.5, 2])!
	assert result.array_equal(expected)
}

fn test_divide_2d() {
	a := from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	b := from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	result := a.divide(b)!
	expected := from_2d([[1, 1, 1, 1], [1, 1, 1, 1]])!
	assert result.array_equal(expected)
}

fn test_divide_2d_scalar() {
	a := from_2d([[1.0, 2, 3, 4], [5.0, 6, 7, 8]])!
	result := a.divide_scalar(2.0)!
	expected := from_2d([[0.5, 1, 1.5, 2], [2.5, 3, 3.5, 4]])!
	assert result.array_equal(expected)
}
