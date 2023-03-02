module vtl

import math

// broadcastable takes two Tensors and either returns a valid
// broadcastable shape or an error
pub fn (a &Tensor[T]) broadcastable[T](b &Tensor[T]) ![]int {
	a_size := a.rank()
	b_size := b.rank()

	if a_size == b_size {
		if broadcast_equal(a.shape, b.shape) {
			return broadcastable_shape(a.shape, b.shape)
		}
	} else {
		if a_size > b_size {
			mut shape := []int{len: a_size - b_size, init: 1}
			shape << b.shape
			if broadcast_equal(a.shape, shape) {
				return broadcastable_shape(a.shape, shape)
			}
		} else {
			mut shape := []int{len: b_size - a_size, init: 1}
			shape << a.shape
			if broadcast_equal(shape, b.shape) {
				return broadcastable_shape(shape, b.shape)
			}
		}
	}
	return error('Shapes ${a.shape} and ${b.shape} are not broadcastable')
}

// broadcast_equal asserts that two shapes can be broadcast
// against each other
fn broadcast_equal(a []int, b []int) bool {
	for i, v in a {
		if !(v == b[i] || v == 1 || b[i] == 1) {
			return false
		}
	}
	return true
}

// broadcastable_shape returns the proper size at each dimension
// of two pre-processed broadcastable shapes
fn broadcastable_shape(a []int, b []int) []int {
	mut result := []int{}
	for i, av in a {
		result << math.max(av, b[i])
	}
	return result
}

// broadcast strides broadcasts the strides of an existing array to
// allow it to be viewed as a compatible shape
fn broadcast_strides(dest_shape []int, src_shape []int, dest_strides []int, src_strides []int) ![]int {
	dims := dest_shape.len
	start := dims - src_shape.len
	mut result := []int{len: dims, init: 0}
	for i := dims - 1; i >= start; i-- {
		t := src_shape[i - start]
		if t == 1 {
			result[i] = 0
		} else if t == dest_shape[i] {
			result[i] = src_strides[i - start]
		} else {
			return error('Cannot broadcast from ${src_shape} to ${dest_shape}')
		}
	}
	return result
}

// broadcast_to broadcasts a Tensor to a compatible shape with no
// data copy
pub fn (t &Tensor[T]) broadcast_to[T](shape []int) !&Tensor[T] {
	if t.shape == shape {
		return t
	}
	size := size_from_shape(shape)
	strides := strides_from_shape(shape, .row_major)
	result_strides := broadcast_strides(shape, t.shape, strides, t.strides)!
	return &Tensor[T]{
		data: t.data
		shape: shape
		size: size
		strides: result_strides
	}
}

fn broadcast_shapes(args ...[]int) []int {
	s0 := args[0]
	mut all_same := true
	for i in 0 .. args.len {
		if s0 != args[i] {
			all_same = false
		}
	}
	if all_same {
		return s0
	}
	mut nd := 0
	for i in 0 .. args.len {
		if args[i].len > nd {
			nd = args[i].len
		}
	}
	mut result := []int{len: nd}
	for i in 0 .. args.len {
		ai := args[i]
		d := nd - ai.len
		mut shape := []int{len: d, init: 1}
		shape << ai
		for j := 0; j < shape.len; j++ {
			if shape[j] > result[j] {
				result[j] = shape[j]
			}
		}
	}
	return result
}

// broadcast2 broadcasts two Tensors against each other
[inline]
pub fn broadcast2[T](a &Tensor[T], b &Tensor[T]) ?(&Tensor[T], &Tensor[T]) {
	shape := a.broadcastable(b)?
	r1 := a.broadcast_to(shape)?
	r2 := b.broadcast_to(shape)?
	return r1, r2
}

// broadcast3 broadcasts three Tensors against each other
[inline]
pub fn broadcast3[T](a &Tensor[T], b &Tensor[T], c &Tensor[T]) ?(&Tensor[T], &Tensor[T], &Tensor[T]) {
	shape := broadcast_shapes(a.shape, b.shape, c.shape)
	r1 := a.broadcast_to(shape)?
	r2 := b.broadcast_to(shape)?
	r3 := c.broadcast_to(shape)?
	return r1, r2, r3
}

// broadcast_n broadcasts N Tensors against each other
[inline]
pub fn broadcast_n[T](ts []&Tensor[T]) ![]&Tensor[T] {
	shapes := ts.map(it.shape)
	shape := broadcast_shapes(...shapes)
	mut result := []&Tensor[T]{cap: ts.len}
	for t in ts {
		result << t.broadcast_to(shape)!
	}
	return result
}
