module la

import vtl

fn correct_axes[T](a &vtl.Tensor[T], b &vtl.Tensor[T], axes_a_ []int, axes_b_ []int) ?([]int, []int) {
	mut equal := true
	mut axes_a := axes_a_.clone()
	mut axes_b := axes_b_.clone()
	if axes_a.len != axes_b.len {
		equal = false
	} else {
		a_shape := a.shape
		a_rank := a.rank()
		b_shape := b.shape
		b_rank := b.rank()
		for k in 0 .. axes_a.len {
			if a_shape[axes_a[k]] != b_shape[axes_b[k]] {
				equal = false
				break
			}
			if axes_a[k] < 0 {
				axes_a[k] += a_rank
			}
			if axes_b[k] < 0 {
				axes_b[k] += b_rank
			}
		}
	}
	if !equal {
		return error('Shape mismatch for sum')
	}
	return axes_a_, axes_b_
}

fn tensordot_output_data[T](a &vtl.Tensor[T], b &vtl.Tensor[T], a_axes_ []int, b_axes_ []int) ([]int, []int, []int, []int, []int) {
	a_shape := a.shape
	a_rank := a.rank()
	b_shape := b.shape
	b_rank := b.rank()
	a_axes, b_axes := correct_axes(a, b, a_axes_, b_axes_) or { panic(err) }
	tmp := vtl.irange(0, a_rank)
	notin := tmp.filter(it !in a_axes)
	mut a_newaxes := notin.clone()
	a_newaxes << a_axes
	mut n2 := 1
	for axis in a_axes {
		n2 *= a_shape[axis]
	}
	firstdim := notin.map(a_shape[it])
	val := vtl.iarray_prod(firstdim)
	a_newshape := [val, n2]
	tmpb := vtl.irange(0, b_rank)
	notinb := tmpb.filter(it !in b_axes)
	mut b_newaxes := b_axes.clone()
	b_newaxes << notinb
	n2 = 1
	for axis in b_axes {
		n2 *= b_shape[axis]
	}
	firstdimb := notin.map(b_shape[it])
	valb := vtl.iarray_prod(firstdimb)
	b_newshape := [n2, valb]
	mut outshape := []int{}
	outshape << firstdim
	outshape << firstdimb
	return outshape, a_newshape, b_newshape, a_newaxes, b_newaxes
}

pub fn tensordot[T](a &vtl.Tensor[T], b &vtl.Tensor[T], a_axes []int, b_axes []int) &vtl.Tensor[T] {
	outshape, a_newshape, b_newshape, a_newaxes, b_newaxes := tensordot_output_data(a,
		b, a_axes, b_axes)
	at := a.transpose(a_newaxes).reshape(a_newshape)
	bt := b.transpose(b_newaxes).reshape(b_newshape)
	res := matmul(at, bt)
	return res.reshape(outshape)
}
