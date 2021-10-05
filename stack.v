module vtl

pub struct AxisData {
	axis int
}

// Stack arrays in sequence vertically (row wise)
pub fn vstack<T>(ts []&Tensor<T>) &Tensor<T> {
	return concatenate<T>(ts, axis: 0)
}

// Stack arrays in sequence horizontally (column wise)
pub fn hstack<T>(ts []&Tensor<T>) &Tensor<T> {
	if ts[0].rank() == 1 {
		return concatenate<T>(ts, axis: 0)
	}
	return concatenate<T>(ts, axis: 1)
}

// Stack arrays in sequence depth wise (along third axis)
pub fn dstack<T>(ts []&Tensor<T>) &Tensor<T> {
	first_tensor := ts[0]
	assert_shape<T>(first_tensor.shape, ts)
	if first_tensor.rank() > 2 {
		panic('dstack was given arrays with more than two dimensions')
	}
	if first_tensor.rank() == 1 {
		next_ts := ts.map(it.reshape<T>([1, it.size, 1]))
		return concatenate<T>(next_ts, axis: 2)
	} else {
		mut next_ts := []&Tensor<T>{cap: ts.len}
		for t in ts {
			mut newshape := t.shape.clone()
			newshape << 1
			next_ts << t.reshape<T>(newshape)
		}
		return concatenate<T>(next_ts, axis: 2)
	}
}

// Stack 1-D arrays as columns into a 2-D array.
pub fn column_stack<T>(ts []&Tensor<T>) &Tensor<T> {
	first_tensor := ts[0]
	assert_shape<T>(first_tensor.shape, ts)

	if first_tensor.rank() > 2 {
		panic('column_stack was given arrays with more than two dimensions')
	}

	if first_tensor.rank() == 1 {
		next_ts := ts.map(it.reshape<T>([it.size, 1]))
		return concatenate<T>(next_ts, axis: 1)
	}

	return concatenate<T>(ts, axis: 1)
}

// Join a sequence of arrays along a new axis.
pub fn stack<T>(ts []&Tensor<T>, data AxisData) &Tensor<T> {
	assert_shape<T>(ts[0].shape, ts)
	expanded := ts.map(expand_dims<T>(it, data))
	return concatenate<T>(expanded, data)
}

// concatenates two Tensors together
pub fn concatenate<T>(ts []&Tensor<T>, data AxisData) &Tensor<T> {
	mut newshape := ts[0].shape.clone()
	// just a check for negative axes, so that negative axes can be inferred.
	axis := clip_axis(data.axis, newshape.len)
	newshape[axis] = 0
	newshape = assert_shape_off_axis<T>(ts, axis, newshape)
	mut ret := new_tensor_like_with_shape<T>(ts[0], newshape)
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
pub fn expand_dims<T>(t &Tensor<T>, data AxisData) &Tensor<T> {
	axis := data.axis
	mut newshape := []int{}
	newaxis := match (axis < 0) {
		true { axis + t.rank() + 1 }
		else { axis }
	}
	newshape << t.shape[..newaxis]
	newshape << 1
	newshape << t.shape[newaxis..]
	return t.reshape<T>(newshape)
}
