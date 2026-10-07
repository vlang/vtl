module vtl

// broadcastable takes two Tensors and either returns a valid
// broadcastable shape or an error
pub fn (a &Tensor[T]) broadcastable[T](b &Tensor[T]) ![]int {
	return broadcast_shapes(a.shape, b.shape) or {
		error('Shapes are not broadcastable')
	}
}

// broadcast_equal checks aligned dimensions using NumPy's compatibility rule.
fn broadcast_equal(a []int, b []int) bool {
	if a.len != b.len {
		return false
	}
	for i, dimension in a {
		if dimension != b[i] && dimension != 1 && b[i] != 1 {
			return false
		}
	}
	return true
}

// broadcast strides broadcasts the strides of an existing array to
// allow it to be viewed as a compatible shape
fn broadcast_strides(dest_shape []int, src_shape []int, src_strides []int) ![]int {
	dims := dest_shape.len
	if dims < src_shape.len {
		return error('Cannot broadcast: target rank is smaller than the source rank')
	}
	start := dims - src_shape.len
	mut result := []int{len: dims, init: 0}
	for i := dims - 1; i >= start; i-- {
		t := src_shape[i - start]
		if dest_shape[i] < 0 {
			return error('Cannot broadcast to a shape with negative dimensions')
		}
		if t == 1 {
			result[i] = 0
		} else if t == dest_shape[i] {
			result[i] = src_strides[i - start]
		} else {
			return error('Cannot broadcast: incompatible dimensions')
		}
	}
	return result
}

// broadcast_to broadcasts a Tensor to a compatible shape with no
// data copy
pub fn (t &Tensor[T]) broadcast_to[T](shape []int) !&Tensor[T] {
	for dimension in shape {
		if dimension < 0 {
			return error('Cannot broadcast to a shape with negative dimensions')
		}
	}
	if shape.len < t.rank() {
		return error('Cannot broadcast: target rank is smaller than the source rank')
	}
	if t.shape == shape {
		return t
	}
	size := size_from_shape(shape)
	result_strides := broadcast_strides(shape, t.shape, t.strides)!
	return &Tensor[T]{
		data:    t.data
		shape:   shape
		size:    size
		strides: result_strides
	}
}

fn broadcast_shapes(args ...[]int) ![]int {
	if args.len == 0 {
		return error('Broadcasting requires at least one shape')
	}
	mut nd := 0
	for shape in args {
		if shape.len > nd {
			nd = shape.len
		}
	}
	mut result := []int{len: nd, init: 1}
	for axis in 0 .. nd {
		for shape in args {
			leading := nd - shape.len
			if axis < leading {
				continue
			}
			dimension := shape[axis - leading]
			if dimension < 0 {
				return error('Shape dimensions must be non-negative')
			}
			if !broadcast_equal([result[axis]], [dimension]) {
				return error('Shapes are not broadcastable')
			}
			if dimension == 1 {
				continue
			}
			if result[axis] == 1 || result[axis] == dimension {
				result[axis] = dimension
			} else {
				return error('Shapes are not broadcastable')
			}
		}
	}
	return result
}

// broadcast2 broadcasts two Tensors against each other

// broadcast2 exposes this operation as part of the public API.

// broadcast2 exposes this operation as part of the public API.
@[inline]
pub fn broadcast2[T](a &Tensor[T], b &Tensor[T]) !(&Tensor[T], &Tensor[T]) {
	shape := a.broadcastable(b)!
	r1 := a.broadcast_to(shape)!
	r2 := b.broadcast_to(shape)!
	return r1, r2
}

// broadcast3 broadcasts three Tensors against each other

// broadcast3 exposes this operation as part of the public API.

// broadcast3 exposes this operation as part of the public API.
@[inline]
pub fn broadcast3[T](a &Tensor[T], b &Tensor[T], c &Tensor[T]) !(&Tensor[T], &Tensor[T], &Tensor[T]) {
	shape := broadcast_shapes(a.shape, b.shape, c.shape)!
	r1 := a.broadcast_to(shape)!
	r2 := b.broadcast_to(shape)!
	r3 := c.broadcast_to(shape)!
	return r1, r2, r3
}

// broadcast_n broadcasts N Tensors against each other

// broadcast_n exposes this operation as part of the public API.

// broadcast_n exposes this operation as part of the public API.
@[inline]
pub fn broadcast_n[T](ts []&Tensor[T]) ![]&Tensor[T] {
	if ts.len == 0 {
		return error('broadcast_n requires at least one tensor')
	}
	shapes := ts.map(it.shape)
	shape := broadcast_shapes(...shapes)!
	mut result := []&Tensor[T]{cap: ts.len}
	for t in ts {
		result << t.broadcast_to(shape)!
	}
	return result
}
