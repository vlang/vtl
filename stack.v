module vtl

// AxisData defines a public data structure for this module.

// AxisData defines a public data structure for this module.
@[params]
pub struct AxisData {
pub:
	axis int
}

// vstack stack arrays in sequence vertically (row wise)
pub fn vstack[T](ts []&Tensor[T]) !&Tensor[T] {
	return concatenate[T](ts, axis: 0)
}

// hstack stacks arrays in sequence horizontally (column wise)
pub fn hstack[T](ts []&Tensor[T]) !&Tensor[T] {
	if ts[0].rank() == 1 {
		return concatenate[T](ts, axis: 0)
	}
	return concatenate[T](ts, axis: 1)
}

// dstack stacks arrays in sequence depth wise (along third axis)
pub fn dstack[T](ts []&Tensor[T]) !&Tensor[T] {
	first_tensor := ts[0]
	assert_shape[T](first_tensor.shape, ts)!
	if first_tensor.rank() > 2 {
		return error('dstack was given arrays with more than two dimensions')
	}
	if first_tensor.rank() == 1 {
		mut next_ts := []&Tensor[T]{cap: ts.len}
		for t in ts {
			next_ts << t.reshape[T]([1, t.size, 1])!
		}
		return concatenate[T](next_ts, axis: 2)
	} else {
		mut next_ts := []&Tensor[T]{cap: ts.len}
		for t in ts {
			mut newshape := t.shape.clone()
			newshape << 1
			next_ts << t.reshape[T](newshape)!
		}
		return concatenate[T](next_ts, axis: 2)
	}
}

// column_stack stacks 1-D arrays as columns into a 2-D array.
pub fn column_stack[T](ts []&Tensor[T]) !&Tensor[T] {
	first_tensor := ts[0]
	assert_shape[T](first_tensor.shape, ts)!

	if first_tensor.rank() > 2 {
		return error('column_stack was given arrays with more than two dimensions')
	}

	if first_tensor.rank() == 1 {
		mut next_ts := []&Tensor[T]{cap: ts.len}
		for t in ts {
			next_ts << t.reshape[T]([t.size, 1])!
		}
		return concatenate[T](next_ts, axis: 1)
	}

	return concatenate[T](ts, axis: 1)
}

// stack join a sequence of arrays along a new axis.
pub fn stack[T](ts []&Tensor[T], data AxisData) !&Tensor[T] {
	assert_shape[T](ts[0].shape, ts)!
	mut expanded := []&Tensor[T]{cap: ts.len}
	for t in ts {
		expanded << t.expand_dims[T](data)!
	}
	return concatenate[T](expanded, data)
}

fn shape_view[T](t &Tensor[T], shape []int, strides []int) &Tensor[T] {
	mut view := t.view()
	view.shape = shape
	view.strides = strides
	return view
}

// unsqueeze adds a dimension of size one to a Tensor
pub fn (t &Tensor[T]) unsqueeze[T](data AxisData) !&Tensor[T] {
	axis := data.axis
	mut newshape := t.shape.clone()
	newaxis := match axis < 0 {
		true { axis + t.rank() + 1 }
		else { axis }
	}

	actual_axis := if newaxis > t.rank() { t.rank() } else { newaxis }
	newshape.insert(actual_axis, 1)
	inserted_stride := if actual_axis < t.rank() { t.strides[actual_axis] } else { 1 }
	mut newstrides := t.strides.clone()
	newstrides.insert(actual_axis, inserted_stride)
	return shape_view[T](t, newshape, newstrides)
}

// squeeze removes dimensions of size one from a Tensor.
// If dim is provided, only that dimension is removed (must be size 1).
// If no dim is provided, all size-1 dimensions are removed.
pub fn (t &Tensor[T]) squeeze[T](data AxisData) !&Tensor[T] {
	mut newshape := []int{}
	mut newstrides := []int{}
	if data.axis < 0 {
		// Remove all size-1 dimensions
		for i, dim in t.shape {
			if dim != 1 {
				newshape << dim
				newstrides << t.strides[i]
			}
		}
	} else {
		// Remove only the specified dimension
		axis := if data.axis >= t.rank() { t.rank() - 1 } else { data.axis }
		for i, dim in t.shape {
			if i != axis || dim != 1 {
				newshape << dim
				newstrides << t.strides[i]
			}
		}
	}
	if newshape.len == 0 {
		newshape = [1]
		newstrides = [1]
	}
	return shape_view[T](t, newshape, newstrides)
}

// concatenate concatenates two Tensors together
pub fn concatenate[T](ts []&Tensor[T], data AxisData) !&Tensor[T] {
	mut newshape := ts[0].shape.clone()
	// just a check for negative axes, so that negative axes can be inferred.
	axis := clip_axis(data.axis, newshape.len)!
	newshape[axis] = 0
	shape := assert_shape_off_axis[T](ts, axis, newshape)!
	mut ret := tensor[T](cast[T](0), shape)
	mut lo := []int{len: shape.len}
	mut hi := shape.clone()
	hi[axis] = 0
	for t in ts {
		if t.shape[axis] != 0 {
			hi[axis] += t.shape[axis]
			mut slice := ret.slice_hilo(lo, hi)!
			slice.assign(t)!
			lo[axis] = hi[axis]
		}
	}
	return ret
}

// expand_dims adds an axis to a Tensor in order to support
// broadcasting operations
pub fn (t &Tensor[T]) expand_dims[T](data AxisData) !&Tensor[T] {
	axis := data.axis
	mut newshape := []int{}
	newaxis := match axis < 0 {
		true { axis + t.rank() + 1 }
		else { axis }
	}

	newshape << t.shape[..newaxis]
	newshape << 1
	newshape << t.shape[newaxis..]
	mut newstrides := []int{}
	newstrides << t.strides[..newaxis]
	newstrides << if newaxis < t.rank() { t.strides[newaxis] } else { 1 }
	newstrides << t.strides[newaxis..]
	return shape_view[T](t, newshape, newstrides)
}
