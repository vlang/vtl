module vtl

// broadcast_to broadcasts a Tensor to a compatible shape with no
// data copy
pub fn (t &VclTensor[T]) broadcast_to[T](shape []int) ?&VclTensor[T] {
	if t.shape == shape {
		return t
	}
	size := size_from_shape(shape)
	strides := strides_from_shape(shape, .row_major)
	result_strides := broadcast_strides(shape, t.shape, strides, t.strides)?
	return &VclTensor[T]{
		data: t.data
		shape: shape
		size: size
		strides: result_strides
	}
}
