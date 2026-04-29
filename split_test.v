module vtl

fn test_array_split() {
	a := seq[f64](9).reshape([3, 3])!
	e1 := from_array([0.0, 1, 2], [1, 3])!
	e2 := from_array([3.0, 4, 5], [1, 3])!
	e3 := from_array([6.0, 7, 8], [1, 3])!
	result := a.array_split(3, 0)!
	assert e1.array_equal(result[0])
	assert e2.array_equal(result[1])
	assert e3.array_equal(result[2])
}

fn test_array_split_expl() {
	a := seq[f64](9).reshape([3, 3])!
	e1 := from_array([0.0, 3, 6], [3, 1])!
	e2 := from_array([1.0, 2, 4, 5, 7, 8], [3, 2])!
	result := a.array_split_expl([1], 1)!
	assert e1.array_equal(result[0])
	assert e2.array_equal(result[1])
}

fn test_split() {
	a := seq[f64](9).reshape([3, 3])!
	e1 := from_array([0.0, 1, 2], [1, 3])!
	e2 := from_array([3.0, 4, 5], [1, 3])!
	e3 := from_array([6.0, 7, 8], [1, 3])!
	result := a.split(3, 0)!
	assert e1.array_equal(result[0])
	assert e2.array_equal(result[1])
	assert e3.array_equal(result[2])
}

fn test_split_expl() {
	a := seq[f64](9).reshape([3, 3])!
	e1 := from_array([0.0, 3, 6], [3, 1])!
	e2 := from_array([1.0, 2, 4, 5, 7, 8], [3, 2])!
	result := a.split_expl([1], 1)!
	assert e1.array_equal(result[0])
	assert e2.array_equal(result[1])
}

fn test_hsplit() {
	a := seq[f64](9).reshape([3, 3])!
	result := a.hsplit(3)!
	expected := a.split(3, 1)!
	assert expected[0].array_equal(result[0])
	assert expected[1].array_equal(result[1])
	assert expected[2].array_equal(result[2])
}

fn test_hsplit_expl() {
	a := seq[f64](9).reshape([3, 3])!
	result := a.hsplit_expl([1, 2])!
	expected := a.split_expl([1, 2], 1)!
	assert expected[0].array_equal(result[0])
	assert expected[1].array_equal(result[1])
}

fn test_vsplit() {
	a := seq[f64](9).reshape([3, 3])!
	result := a.vsplit(3)!
	expected := a.split(3, 0)!
	assert expected[0].array_equal(result[0])
	assert expected[1].array_equal(result[1])
	assert expected[2].array_equal(result[2])
}

fn test_vsplit_expl() {
	a := seq[f64](9).reshape([3, 3])!
	result := a.vsplit_expl([1, 2])!
	expected := a.split_expl([1, 2], 0)!
	assert expected[0].array_equal(result[0])
	assert expected[1].array_equal(result[1])
}

fn test_dsplit() {
	a := seq[f64](27).reshape([3, 3, 3])!
	result := a.dsplit(3)!
	expected := a.split(3, 2)!
	assert expected[0].array_equal(result[0])
	assert expected[1].array_equal(result[1])
	assert expected[2].array_equal(result[2])
}

fn test_dsplit_expl() {
	a := seq[f64](27).reshape([3, 3, 3])!
	result := a.dsplit_expl([1, 2])!
	expected := a.split_expl([1, 2], 2)!
	assert expected[0].array_equal(result[0])
	assert expected[1].array_equal(result[1])
}
