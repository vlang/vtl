import vtl.num

fn get_input() num.NdArray {
	return num.from_int([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])
}

fn test_sum() {
	a := get_input()
	result := a.sum()
	assert result == 45
}

fn test_prod() {
	a := get_input()
	result := num.prod(a)
	assert result == 362880
}

fn test_mean() {
	a := get_input()
	result := num.mean(a)
	assert result == 5
}

fn test_max() {
	a := get_input()
	result := a.max()
	assert result == 9
}

fn test_min() {
	a := get_input()
	result := a.min()
	assert result == 1
}

fn test_ptp() {
	a := get_input()
	result := a.ptp()
	assert result == 8
}

fn test_axis_sum() {
	a := get_input()
	result := a.sum_axis(0)
	expected := num.from_int_1d([12, 15, 18])
	assert num.allclose(result, expected)
}

fn test_axis_sum_keepdims() {
	a := get_input()
	result := num.sum_axis_dims(a, 0)
	expected := num.from_int([12, 15, 18], [1, 3])
	assert num.allclose(result, expected)
}

fn test_axis_mean() {
	a := get_input()
	result := num.mean_axis(a, 0)
	expected := num.from_int_1d([4, 5, 6])
	assert num.allclose(result, expected)
}

fn test_axis_mean_keepdims() {
	a := get_input()
	result := num.mean_axis_dims(a, 0)
	expected := num.from_int([4, 5, 6], [1, 3])
	assert num.allclose(result, expected)
}

fn test_axis_minimum() {
	a := get_input()
	result := a.min_axis(0)
	expected := num.from_int_1d([1, 2, 3])
	assert num.allclose(result, expected)
}

fn test_axis_minimum_keepdims() {
	a := get_input()
	result := num.min_axis_dims(a, 0)
	expected := num.from_int([1, 2, 3], [1, 3])
	assert num.allclose(result, expected)
}

fn test_axis_maximum() {
	a := get_input()
	result := a.max_axis(0)
	expected := num.from_int_1d([7, 8, 9])
	assert num.allclose(result, expected)
}

fn test_axis_maximum_keepdims() {
	a := get_input()
	result := num.max_axis_dims(a, 0)
	expected := num.from_int([7, 8, 9], [1, 3])
	assert num.allclose(result, expected)
}
