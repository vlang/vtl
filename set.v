module vtl

import math

// isin returns a boolean tensor indicating whether each element occurs in
// test_elements. It preserves the shape and logical iteration order of the
// input, including for non-contiguous views.
pub fn isin[T](elements &Tensor[T], test_elements &Tensor[T]) &Tensor[bool] {
	mut sorted := test_elements.to_array()
	sorted.sort_with_compare(fn [T](a &T, b &T) int {
		return compare_isin_values[T](*a, *b)
	})
	mut result := empty[bool](elements.shape, memory: .row_major)
	mut iter := elements.iterator[T]()
	mut linear := 0
	for {
		value, _ := iter.next() or { break }
		result.set_nth(linear, sorted_contains[T](sorted, value))
		linear++
	}
	return result
}

fn sorted_contains[T](sorted []T, value T) bool {
	$if T is f32 || T is f64 {
		if math.is_nan(f64(value)) {
			return false
		}
	}
	mut low := 0
	mut high := sorted.len
	for low < high {
		middle := low + (high - low) / 2
		if compare_isin_values[T](sorted[middle], value) < 0 {
			low = middle + 1
		} else {
			high = middle
		}
	}
	return low < sorted.len && sorted[low] == value
}

fn compare_isin_values[T](a T, b T) int {
	$if T is bool {
		if a == b {
			return 0
		}
		return if a { 1 } else { -1 }
	} $else {
		return compare_sort_values[T](a, b)
	}
}
