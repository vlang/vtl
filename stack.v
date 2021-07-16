module vtl

pub struct AxisData {
	axis int
}

// Stack arrays in sequence vertically (row wise)
pub fn vstack(ts []&Tensor) &Tensor {
	return concatenate(ts, axis: 0)
}

// Stack arrays in sequence horizontally (column wise)
pub fn hstack(ts []&Tensor) &Tensor {
	if ts[0].rank() == 1 {
		return concatenate(ts, axis: 0)
	}
	return concatenate(ts, axis: 1)
}

// Stack arrays in sequence depth wise (along third axis)
pub fn dstack(ts []&Tensor) &Tensor {
	first_tensor := ts[0]
	assert_shape(first_tensor.shape, ts)
	if first_tensor.rank() > 2 {
		panic('dstack was given arrays with more than two dimensions')
	}
	if first_tensor.rank() == 1 {
		next_ts := ts.map(it.reshape([1, it.size, 1]))
		return concatenate(next_ts, axis: 2)
	} else {
		mut next_ts := []&Tensor{cap: ts.len}
		for t in ts {
			mut newshape := t.shape.clone()
			newshape << 1
			next_ts << t.reshape(newshape)
		}
		return concatenate(next_ts, axis: 2)
	}
}

// Stack 1-D arrays as columns into a 2-D array.
pub fn column_stack(ts []&Tensor) &Tensor {
	first_tensor := ts[0]
	assert_shape(first_tensor.shape, ts)

	if first_tensor.rank() > 2 {
		panic('column_stack was given arrays with more than two dimensions')
	}

	if first_tensor.rank() == 1 {
		next_ts := ts.map(it.reshape([it.size, 1]))
		return concatenate(next_ts, axis: 1)
	}

	return concatenate(ts, axis: 1)
}

// Join a sequence of arrays along a new axis.
pub fn stack(ts []&Tensor, data AxisData) &Tensor {
	assert_shape(ts[0].shape, ts)
	expanded := ts.map(expand_dims(it, data))
	return concatenate(expanded, data)
}

// concatenates two Tensors together
pub fn concatenate(ts []&Tensor, data AxisData) &Tensor {
	mut newshape := ts[0].shape.clone()
	// just a check for negative axes, so that negative axes can be inferred.
	axis := clip_axis(data.axis, newshape.len)
	newshape[axis] = 0
	newshape = assert_shape_off_axis(ts, axis, newshape)
	mut ret := new_tensor_like_with_shape(ts[0], newshape)
	mut lo := []int{len: newshape.len}
	mut hi := newshape.clone()
	hi[axis] = 0
	for t in ts {
		if t.shape[axis] != 0 {
			hi[axis] += t.shape[axis]
			mut slice := ret.slice_hilo(lo, hi)
			slice.assign(t)
			lo[axis] = hi[axis]
		}
	}
	return ret
}

// expand_dims adds an axis to a Tensor in order to support
// broadcasting operations
pub fn expand_dims(t Tensor, data AxisData) &Tensor {
	axis := data.axis
	mut newshape := []int{}
	newaxis := match (axis < 0) {
		true { axis + t.rank() + 1 }
		else { axis }
	}
	newshape << t.shape[..newaxis]
	newshape << 1
	newshape << t.shape[newaxis..]
	return t.reshape(newshape)
}
