module vtl

// concatenates two Tensors together
pub fn concatenate(ts []Tensor, axis int) Tensor {
	mut newshape := ts[0].shape
	newshape[axis] = 0
	newshape = assert_shape_off_axis(ts, axis, newshape)
	mut ret := new_tensor_like_with_shape(ts[0], newshape)
	mut lo := []int{len: newshape.len}
	mut hi := newshape
	hi[axis] = 0
	for t in ts {
		if t.shape[axis] != 0 {
			hi[axis] += t.shape[axis]
			ret = ret.slice([lo, hi])
                        ret.assign(t)
			lo[axis] = hi[axis]
		}
	}
	return ret
}
