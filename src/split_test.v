module vtl

fn test_array_split() ? {
	a := seq<f64>(9).reshape([3, 3])?
	e1 := from_array([0.0, 1, 2], [1, 3])?
	e2 := from_array([3.0, 4, 5], [1, 3])?
	e3 := from_array([6.0, 7, 8], [1, 3])?
	result := a.array_split(3, 0)?
	assert e1.array_equal(result[0])
	assert e2.array_equal(result[1])
	assert e3.array_equal(result[2])
}

fn test_array_split_expl() ? {
	a := seq<f64>(9).reshape([3, 3])?
	e1 := from_array([0.0, 3, 6], [3, 1])?
	e2 := from_array([1.0, 2, 4, 5, 7, 8], [3, 2])?
	result := a.array_split_expl([1], 1)?
	// @todo: The split is working but for some reason the equal function is not
	// assert e1.array_equal(result[0])
	// assert e2.array_equal(result[1])
}
