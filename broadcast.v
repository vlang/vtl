module vtl

// broadcastable takes two Tensors and either returns a valid
// broadcastable shape or panics
pub fn broadcastable(a Tensor, b Tensor) []int {
	if a.rank() == b.rank() {
		if broadcast_equal(a.shape, b.shape) {
			return broadcastable_shape(a.shape, b.shape)
		}
	} else {
		if a.rank() > b.rank() {
			mut b_shape := []int{len: a.rank() - b.rank(), init: int(1)}
			b_shape << b.shape
			if broadcast_equal(a.shape, b_shape) {
				return broadcastable_shape(a.shape, b_shape)
			}
		} else {
			mut a_shape := []int{len: b.rank() - a.rank(), init: int(1)}
			a_shape << a.shape
			if broadcast_equal(a_shape, b.shape) {
				return broadcastable_shape(a_shape, b.shape)
			}
		}
	}
	panic('Shapes $a.shape and $b.shape are not broadcastable')
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
		if av > b[i] {
			result << av
		} else {
			result << b[i]
		}
	}
	return result
}

// broadcast strides broadcasts the strides of an existing array to
// allow it to be viewed as a compatible shape
fn broadcast_strides(dshape []int, sshape []int, dstrides []int, sstrides []int) []int {
	d := dshape.len
	s := d - sshape.len
	mut result := []int{len: d, init: 0}
	for i := d - 1; i >= s; i-- {
		t := sshape[i - s]
		if t == 1 {
			result[i] = 0
		} else if t == dshape[i] {
			result[i] = sstrides[i - s]
		} else {
			panic('Cannot broadcast from $sshape to $dshape')
		}
	}
	return result
}

// broadcast_to broadcasts a Tensor to a compatible shape with no
// data copy
pub fn (t Tensor) broadcast_to(shape []int) Tensor {
	if t.shape == shape {
		return t
	}
	size := size_from_shape(shape)
	strides := strides_from_shape(shape, .rowmajor)
	result_strides := broadcast_strides(shape, t.shape, strides, t.strides)
	return Tensor{
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
		mut shape := []int{len: d, init: int(1)}
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
fn broadcast2(a Tensor, b Tensor) (Tensor, Tensor) {
	shape := broadcast_shapes(a.shape, b.shape)
	return a.broadcast_to(shape), b.broadcast_to(shape)
}

// broadcast3 broadcasts three Tensors against each other
[inline]
fn broadcast3(a Tensor, b Tensor, c Tensor) (Tensor, Tensor, Tensor) {
	shape := broadcast_shapes(a.shape, b.shape, c.shape)
	return a.broadcast_to(shape), b.broadcast_to(shape), c.broadcast_to(shape)
}
