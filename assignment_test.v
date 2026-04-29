module vtl

fn test_set() {
	mut t := from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 5])!
	t.set([0, 0], 16)
	t.set([0, 1], 17)
	t.set([0, 2], 18)
	t.set([0, 3], 19)
	t.set([0, 4], 20)
	t.set([1, 0], 11)
	t.set([1, 1], 12)
	t.set([1, 2], 13)
	t.set([1, 3], 14)
	t.set([1, 4], 15)
	expected := from_array([16, 17, 18, 19, 20, 11, 12, 13, 14, 15], [2, 5])!
	assert t.array_equal(expected)
}

fn test_set_nth() {
	mut t := from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 5])!
	t.set_nth(0, 16)
	t.set_nth(1, 17)
	t.set_nth(2, 18)
	t.set_nth(3, 19)
	t.set_nth(4, 20)
	t.set_nth(5, 11)
	t.set_nth(6, 12)
	t.set_nth(7, 13)
	t.set_nth(8, 14)
	t.set_nth(9, 15)
	expected := from_array([16, 17, 18, 19, 20, 11, 12, 13, 14, 15], [2, 5])!
	assert t.array_equal(expected)
}

fn test_fill() {
	mut t := from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 5])!
	t.fill(-1)
	expected := from_array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [2, 5])!
	assert t.array_equal(expected)
}

fn test_assign() {
	mut t := from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 5])!
	a := from_array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [2, 5])!
	t.assign(a)!
	assert t.array_equal(a)
	b := from_1d([21, 22, 23, 24, 25])!
	t.assign(b)!
	expected := from_array([21, 22, 23, 24, 25, 21, 22, 23, 24, 25], [2, 5])!
	assert t.array_equal(expected)
}
