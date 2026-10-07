module vtl

// where selects elements from x or y according to condition, broadcasting all
// three tensors to their common shape. True condition elements select x.
pub fn where[T](condition &Tensor[bool], x &Tensor[T], y &Tensor[T]) !&Tensor[T] {
	shape := broadcast_shapes(condition.shape, x.shape, y.shape)!
	condition_view := condition.broadcast_to(shape)!
	x_view := x.broadcast_to(shape)!
	y_view := y.broadcast_to(shape)!
	mut condition_iter := condition_view.iterator[bool]()
	mut x_iter := x_view.iterator[T]()
	mut y_iter := y_view.iterator[T]()
	mut result := empty[T](shape, memory: .row_major)
	for {
		condition_value, index := condition_iter.next() or { break }
		x_value, _ := x_iter.next() or { break }
		y_value, _ := y_iter.next() or { break }
		result.set(index, if condition_value { x_value } else { y_value })
	}
	return result
}

// choose selects values from choices using integer indices. Choices broadcast
// to one common shape, and indices broadcast to that shape. Each index must be
// in the range [0, choices.len).
pub fn choose[T](indices &Tensor[int], choices []&Tensor[T]) !&Tensor[T] {
	if choices.len == 0 {
		return error('choose requires at least one choice tensor')
	}
	mut shapes := [][]int{cap: choices.len + 1}
	shapes << indices.shape
	for choice in choices {
		shapes << choice.shape
	}
	shape := broadcast_shapes(...shapes)!
	broadcast_indices := indices.broadcast_to(shape)!
	mut broadcast_choices := []&Tensor[T]{cap: choices.len}
	for choice in choices {
		broadcast_choices << choice.broadcast_to(shape)!
	}
	mut result := empty[T](shape, memory: .row_major)
	mut index_iter := broadcast_indices.iterator[int]()
	for {
		choice_index, output_index := index_iter.next() or { break }
		if choice_index < 0 || choice_index >= broadcast_choices.len {
			return error('choose index ${choice_index} at ${output_index} is out of bounds for ${broadcast_choices.len} choices')
		}
		result.set(output_index, broadcast_choices[choice_index].get[T](output_index))
	}
	return result
}
