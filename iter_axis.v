module vtl

// TensorAxisIterator is the core iterator for axis-wise operations.
// Stores a copy of the shape and strides reduced along a given axis
[heap]
pub struct TensorAxisIterator<T> {
pub:
	tensor &Tensor<T>
pub mut:
	shape     []int
	strides   []int
	axis      int
	inc       int
	iteration int
	pos       int
}

// axis_iterator returns an iterator over the axis of a Tensor, commonly
// used for reduction operations along an axis.
pub fn (t &Tensor<T>) axis_iterator<T>(axis int) &TensorAxisIterator<T> {
	mut shape := t.shape.clone()
	mut strides := t.strides.clone()

	shape.delete(axis)
	strides.delete(axis)

	return &TensorAxisIterator<T>{
		tensor: t
		shape: shape
		strides: strides
		axis: axis
		inc: t.strides[axis]
	}
}

// axis returns an iterator over the axis of a Tensor, commonly
// used for reduction operations along an axis. This iterator
// keeps the axis dimension as size 1 instead of removing it
pub fn (t &Tensor<T>) axis_with_dims_iterator<T>(axis int) &TensorAxisIterator<T> {
	mut shape := t.shape.clone()
	mut strides := t.strides.clone()

	shape[axis] = 1
	strides[axis] = 0

	return &TensorAxisIterator<T>{
		tensor: t
		shape: shape
		strides: strides
		axis: axis
		inc: t.strides[axis]
	}
}

// next calls the iteration type for a given iterator
// which is either flat or strided and returns a Num containing the current value
[inline]
pub fn (mut s TensorAxisIterator<T>) next<T>() ?(T, int) {
	if s.iteration >= s.tensor.shape[s.axis] {
		return none
	}
	defer {
		s.iteration++
		s.pos += s.inc
	}
	val := s.tensor.get_nth(s.pos)
	return val, s.pos
}
