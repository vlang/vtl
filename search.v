module vtl

// SearchSide selects which side of equal values receives an insertion index.
pub enum SearchSide {
	left
	right
}

// searchsorted returns insertion positions for values in a sorted 1-D tensor.
// The output has the same shape as values. The input must be non-decreasing.
pub fn searchsorted[T](sorted &Tensor[T], values &Tensor[T], side SearchSide) !&Tensor[int] {
	if sorted.rank() != 1 {
		return error('searchsorted: sorted values must be one-dimensional')
	}
	for i in 1 .. sorted.size {
		if compare_sort_values[T](sorted.get_nth[T](i - 1), sorted.get_nth[T](i)) > 0 {
			return error('searchsorted: input must be sorted in ascending order')
		}
	}
	mut result := empty[int](values.shape, memory: .row_major)
	for i in 0 .. values.size {
		insertion_index := sorted_insertion_index[T](sorted, values.get_nth[T](i), side, false)
		result.set(values.nth_index(i), insertion_index)
	}
	return result
}

// digitize assigns values to monotonic bins. Increasing and decreasing edge
// arrays are supported; `right` selects which interval owns an edge value.
pub fn digitize[T](values &Tensor[T], bins &Tensor[T], right bool) !&Tensor[int] {
	if bins.rank() != 1 {
		return error('digitize: bins must be one-dimensional')
	}
	mut ascending := true
	mut descending := true
	for i in 1 .. bins.size {
		comparison := compare_sort_values[T](bins.get_nth[T](i - 1), bins.get_nth[T](i))
		if comparison > 0 {
			ascending = false
		}
		if comparison < 0 {
			descending = false
		}
	}
	if !ascending && !descending {
		return error('digitize: bins must be monotonic')
	}
	mut result := empty[int](values.shape, memory: .row_major)
	for i in 0 .. values.size {
		side := if right { SearchSide.left } else { SearchSide.right }
		insertion_index := sorted_insertion_index[T](bins, values.get_nth[T](i), side,
			!ascending && descending)
		result.set(values.nth_index(i), insertion_index)
	}
	return result
}

fn sorted_insertion_index[T](bins &Tensor[T], value T, side SearchSide, descending bool) int {
	mut low := 0
	mut high := bins.size
	for low < high {
		mid := low + (high - low) / 2
		comparison := compare_sort_values[T](bins.get_nth[T](mid), value)
		should_advance := if descending {
			if side == .right { comparison > 0 } else { comparison >= 0 }
		} else {
			if side == .right { comparison <= 0 } else { comparison < 0 }
		}
		if should_advance {
			low = mid + 1
		} else {
			high = mid
		}
	}
	return low
}
