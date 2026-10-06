module vtl

// partition returns a copy with the value at kth in its sorted position and
// every preceding value less than or equal to it. Remaining values are
// unspecified, matching NumPy's default last-axis behavior.
pub fn partition[T](t &Tensor[T], kth int) !&Tensor[T] {
	return partition_axis[T](t, [kth], -1)
}

// partition_axis partially sorts each one-dimensional slice along axis. kth
// positions are local to each slice; multiple positions can be requested in
// one pass without fully sorting the data.
pub fn partition_axis[T](t &Tensor[T], kths []int, axis int) !&Tensor[T] {
	axis_index := normalize_sort_axis[T](t, axis)!
	if kths.len == 0 {
		return error('partition_axis: at least one kth position is required')
	}
	axis_size := t.shape[axis_index]
	for kth in kths {
		if kth < 0 || kth >= axis_size {
			return error('partition_axis: kth ${kth} out of bounds for axis of size ${axis_size}')
		}
	}
	mut result := empty[T](t.shape, memory: .row_major)
	if t.size == 0 {
		return result
	}
	line_count := t.size / axis_size
	mut index := []int{len: t.rank()}
	mut values := []T{len: axis_size}
	for line in 0 .. line_count {
		decode_sort_line(line, t.shape, axis_index, mut index)
		for position in 0 .. axis_size {
			index[axis_index] = position
			values[position] = t.get[T](index)
		}
		for kth in kths {
			select_value[T](mut values, kth)
		}
		for position, value in values {
			index[axis_index] = position
			result.set(index, value)
		}
	}
	return result
}

// argpartition returns local indices that place kth at its sorted position
// along the last axis. The order of the other indices is unspecified.
pub fn argpartition[T](t &Tensor[T], kth int) !&Tensor[int] {
	return argpartition_axis[T](t, [kth], -1)
}

// argpartition_axis partially orders local indices along axis. The returned
// tensor has the same shape as t and supports multiple kth positions.
pub fn argpartition_axis[T](t &Tensor[T], kths []int, axis int) !&Tensor[int] {
	axis_index := normalize_sort_axis[T](t, axis)!
	if kths.len == 0 {
		return error('argpartition_axis: at least one kth position is required')
	}
	axis_size := t.shape[axis_index]
	for kth in kths {
		if kth < 0 || kth >= axis_size {
			return error('argpartition_axis: kth ${kth} out of bounds for axis of size ${axis_size}')
		}
	}
	mut result := empty[int](t.shape, memory: .row_major)
	if t.size == 0 {
		return result
	}
	line_count := t.size / axis_size
	mut index := []int{len: t.rank()}
	mut values := []T{len: axis_size}
	mut positions := []int{len: axis_size}
	for line in 0 .. line_count {
		decode_sort_line(line, t.shape, axis_index, mut index)
		for position in 0 .. axis_size {
			index[axis_index] = position
			values[position] = t.get[T](index)
			positions[position] = position
		}
		for kth in kths {
			select_positions[T](values, mut positions, kth)
		}
		for position, source_index in positions {
			index[axis_index] = position
			result.set(index, source_index)
		}
	}
	return result
}

fn select_value[T](mut values []T, kth int) {
	mut low := 0
	mut high := values.len
	for high - low > 1 {
		middle := low + (high - low) / 2
		pivot := selection_median[T](values[low], values[middle], values[high - 1])
		mut less := low
		mut current := low
		mut greater := high
		for current < greater {
			comparison := compare_selection_values[T](values[current], pivot)
			if comparison < 0 {
				values[less], values[current] = values[current], values[less]
				less++
				current++
			} else if comparison > 0 {
				greater--
				values[current], values[greater] = values[greater], values[current]
			} else {
				current++
			}
		}
		if kth < less {
			high = less
		} else if kth >= greater {
			low = greater
		} else {
			return
		}
	}
}

fn select_positions[T](values []T, mut positions []int, kth int) {
	mut low := 0
	mut high := positions.len
	for high - low > 1 {
		middle := low + (high - low) / 2
		pivot := selection_median[T](values[positions[low]], values[positions[middle]],
			values[positions[high - 1]])
		mut less := low
		mut current := low
		mut greater := high
		for current < greater {
			comparison := compare_selection_values[T](values[positions[current]], pivot)
			if comparison < 0 {
				positions[less], positions[current] = positions[current], positions[less]
				less++
				current++
			} else if comparison > 0 {
				greater--
				positions[current], positions[greater] = positions[greater], positions[current]
			} else {
				current++
			}
		}
		if kth < less {
			high = less
		} else if kth >= greater {
			low = greater
		} else {
			return
		}
	}
}

fn selection_median[T](a T, b T, c T) T {
	if compare_selection_values[T](a, b) > 0 {
		if compare_selection_values[T](b, c) > 0 {
			return b
		}
		return if compare_selection_values[T](a, c) > 0 { c } else { a }
	}
	if compare_selection_values[T](a, c) > 0 {
		return a
	}
	return if compare_selection_values[T](b, c) > 0 { c } else { b }
}

fn compare_selection_values[T](a T, b T) int {
	$if T is bool {
		if a == b {
			return 0
		}
		return if a { 1 } else { -1 }
	} $else {
		return compare_sort_values[T](a, b)
	}
}
