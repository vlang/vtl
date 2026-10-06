module vtl

// SearchSide selects which side of equal values receives an insertion index.
pub enum SearchSide {
	left
	right
}

// searchsorted returns insertion positions for values in an ascending 1-D
// tensor. The input is assumed to be sorted; each query uses binary search.
pub fn searchsorted[T](sorted &Tensor[T], values &Tensor[T], side SearchSide) !&Tensor[int] {
	if sorted.rank() != 1 {
		return error('searchsorted: sorted values must be one-dimensional')
	}
	mut result := empty[int](values.shape, memory: .row_major)
	for i in 0 .. values.size {
		insertion_index := sorted_insertion_index[T](sorted, values.get_nth[T](i), side, false)
		result.set(values.nth_index(i), insertion_index)
	}
	return result
}

// searchsorted_descending returns insertion positions for values in a
// descending 1-D tensor. The input is assumed to be sorted; each query uses
// binary search. The output has the same shape as values.
pub fn searchsorted_descending[T](sorted &Tensor[T], values &Tensor[T], side SearchSide) !&Tensor[int] {
	if sorted.rank() != 1 {
		return error('searchsorted_descending: sorted values must be one-dimensional')
	}
	mut result := empty[int](values.shape, memory: .row_major)
	for i in 0 .. values.size {
		insertion_index := sorted_insertion_index[T](sorted, values.get_nth[T](i), side, true)
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
		side := if !ascending && descending {
			if right { SearchSide.right } else { SearchSide.left }
		} else {
			if right { SearchSide.left } else { SearchSide.right }
		}
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
			if side == .right { comparison >= 0 } else { comparison > 0 }
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
