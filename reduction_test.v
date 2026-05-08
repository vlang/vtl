module vtl

fn test_argmax_axis_0() {
	t := from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.argmax_axis(0)!
	// along axis 0: col0 max at row1 (3>1), col1 max at row0 (5>2)
	assert result.get_nth(0) == 1
	assert result.get_nth(1) == 0
}

fn test_argmax_axis_1() {
	t := from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.argmax_axis(1)!
	// along axis 1: row0 max at col1 (5>1), row1 max at col0 (3>2)
	assert result.get_nth(0) == 1
	assert result.get_nth(1) == 0
}

fn test_argmin_axis_0() {
	t := from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.argmin_axis(0)!
	// along axis 0: col0 min at row0 (1<3), col1 min at row1 (2<5)
	assert result.get_nth(0) == 0
	assert result.get_nth(1) == 1
}

fn test_argmin_axis_1() {
	t := from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.argmin_axis(1)!
	// along axis 1: row0 min at col0 (1<5), row1 min at col1 (2<3)
	assert result.get_nth(0) == 0
	assert result.get_nth(1) == 1
}

fn test_argmax_flat() {
	t := from_1d([1.0, 7.0, 3.0, 5.0])!
	result := t.argmax(0)!
	assert result.get_nth(0) == 1
}

fn test_argmin_flat() {
	t := from_1d([4.0, 2.0, 6.0, 1.0])!
	result := t.argmin(0)!
	assert result.get_nth(0) == 3
}

fn test_max_axis_1() {
	t := from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.max_axis(1)!
	assert result.get_nth(0) == f64(5)
	assert result.get_nth(1) == f64(3)
}

fn test_min_axis_1() {
	t := from_2d([[1.0, 5.0], [3.0, 2.0]])!
	result := t.min_axis(1)!
	assert result.get_nth(0) == f64(1)
	assert result.get_nth(1) == f64(2)
}

fn test_cumsum() {
	t := from_1d([1.0, 2.0, 3.0, 4.0])!
	result := t.cumsum(0)!
	assert result.get_nth(0) == f64(1)
	assert result.get_nth(1) == f64(3)
	assert result.get_nth(2) == f64(6)
	assert result.get_nth(3) == f64(10)
}

fn test_cumprod() {
	t := from_1d([1.0, 2.0, 3.0, 4.0])!
	result := t.cumprod(0)!
	assert result.get_nth(0) == f64(1)
	assert result.get_nth(1) == f64(2)
	assert result.get_nth(2) == f64(6)
	assert result.get_nth(3) == f64(24)
}
