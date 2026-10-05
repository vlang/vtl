module vtl

// where selects elements from x or y according to condition, broadcasting all
// three tensors to their common shape. True condition elements select x.
pub fn where[T](condition &Tensor[bool], x &Tensor[T], y &Tensor[T]) !&Tensor[T] {
	shape := broadcast_shapes(condition.shape, x.shape, y.shape)
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
