module vtl

// [inline]
pub fn (t Tensor) rank() int {
	return t.shape.len
}
