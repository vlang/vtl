import vtl.num

fn test_seq() {
	expected := num.from_int_1d([0, 1, 2, 3, 4])
	result := num.seq(5)
	assert num.allclose(expected, result)
}

fn test_seq_between() {
	expected := num.from_int_1d([3, 4, 5, 6, 7, 8])
	result := num.seq_between(3, 9)
	assert num.allclose(expected, result)
}

fn test_linspace() {
	expected := num.seq(11)
	result := num.linspace(0, 5, 11)
	assert num.allclose(num.divide_as(expected, 2), result)
}

fn test_logspace() {
	expected := num.from_f32_1d([f32(1.0), 1.77827941, 3.16227766, 5.62341325, 10.])
	result := num.logspace(0, 1, 5)
	assert num.allclose(expected, result)
}
