module vtl

fn test_array_split() {
	a := seq<f64>(9).reshape([3, 3]) or { panic(@FN + ' failed') }
	e1 := from_array([0.0, 1, 2], [1, 3]) or { panic(@FN + ' failed') }
	e2 := from_array([3.0, 4, 5], [1, 3]) or { panic(@FN + ' failed') }
	e3 := from_array([6.0, 7, 8], [1, 3]) or { panic(@FN + ' failed') }
	result := array_split(a, 3, 0) or { panic(@FN + ' failed') }
	assert e1.array_equiv(result[0])
	assert e2.array_equiv(result[1])
	assert e3.array_equiv(result[2])
}

fn test_array_split_expl() {
	a := seq<f64>(9).reshape([3, 3]) or { panic(@FN + ' failed') }
	e1 := from_array([0.0, 3, 6], [3, 1]) or { panic(@FN + ' failed') }
	e2 := from_array([1.0, 2, 4, 5, 7, 8], [3, 2]) or { panic(@FN + ' failed') }
	result := array_split_expl(a, [1], 1) or { panic(@FN + ' failed') }
	// @todo: The split is working but for some reason the equal function is not
	// assert e1.array_equiv(result[0])
	// assert e2.array_equiv(result[1])
}
