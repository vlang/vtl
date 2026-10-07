module vtl

// kron returns the Kronecker product of two tensors. The lower-rank input is
// left-padded with singleton dimensions, matching NumPy's rank semantics.
pub fn kron[T](a &Tensor[T], b &Tensor[T]) !&Tensor[T] {
	rank := if a.rank() > b.rank() { a.rank() } else { b.rank() }
	a_leading := rank - a.rank()
	b_leading := rank - b.rank()
	mut a_shape := []int{len: a_leading, init: 1}
	a_shape << a.shape
	mut b_shape := []int{len: b_leading, init: 1}
	b_shape << b.shape
	mut shape := []int{cap: rank}
	mut size := 1
	for axis in 0 .. rank {
		a_dimension := a_shape[axis]
		b_dimension := b_shape[axis]
		if a_dimension > 0 && b_dimension > max_int / a_dimension {
			return error('Kronecker product shape overflows the maximum tensor size')
		}
		dimension := a_dimension * b_dimension
		if dimension > 0 && size > max_int / dimension {
			return error('Kronecker product size overflows the maximum tensor size')
		}
		shape << dimension
		size *= dimension
	}
	mut result := empty[T](shape, memory: .row_major)
	mut coordinate := []int{len: rank}
	for flat_index in 0 .. size {
		decode_flat_coordinate(flat_index, shape, mut coordinate)
		mut a_offset := 0
		mut b_offset := 0
		for axis in 0 .. rank {
			a_index := coordinate[axis] / b_shape[axis]
			b_index := coordinate[axis] % b_shape[axis]
			if axis >= a_leading {
				a_offset += a_index * a.strides[axis - a_leading]
			}
			if axis >= b_leading {
				b_offset += b_index * b.strides[axis - b_leading]
			}
		}
		$if T is bool {
			result.data.data[flat_index] = a.data.data[a_offset] && b.data.data[b_offset]
		} $else {
			result.data.data[flat_index] = a.data.data[a_offset] * b.data.data[b_offset]
		}
	}
	return result
}
