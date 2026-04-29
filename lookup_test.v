module vtl

fn test_get() {
	t := from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.get([0, 0]) == 1
	assert t.get([0, 1]) == 2
	assert t.get([1, 0]) == 3
	assert t.get([1, 1]) == 4
	assert t.get([2, 0]) == 5
	assert t.get([2, 1]) == 6
	assert t.get([3, 0]) == 7
	assert t.get([3, 1]) == 8
	assert t.get([4, 0]) == 9
	assert t.get([4, 1]) == 10
}

fn test_get_nth() {
	t := from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.get_nth(0) == 1
	assert t.get_nth(1) == 2
	assert t.get_nth(2) == 3
	assert t.get_nth(3) == 4
	assert t.get_nth(4) == 5
	assert t.get_nth(5) == 6
	assert t.get_nth(6) == 7
	assert t.get_nth(7) == 8
	assert t.get_nth(8) == 9
	assert t.get_nth(9) == 10
}

fn test_offset_index() {
	t := from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.offset_index([0, 0]) == 0
	assert t.offset_index([0, 1]) == 1
	assert t.offset_index([1, 0]) == 2
	assert t.offset_index([1, 1]) == 3
	assert t.offset_index([2, 0]) == 4
	assert t.offset_index([2, 1]) == 5
	assert t.offset_index([3, 0]) == 6
	assert t.offset_index([3, 1]) == 7
	assert t.offset_index([4, 0]) == 8
	assert t.offset_index([4, 1]) == 9
}

fn test_nth_index() {
	t := from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.nth_index(0) == [0, 0]
	assert t.nth_index(1) == [0, 1]
	assert t.nth_index(2) == [1, 0]
	assert t.nth_index(3) == [1, 1]
	assert t.nth_index(4) == [2, 0]
	assert t.nth_index(5) == [2, 1]
	assert t.nth_index(6) == [3, 0]
	assert t.nth_index(7) == [3, 1]
	assert t.nth_index(8) == [4, 0]
	assert t.nth_index(9) == [4, 1]
}
