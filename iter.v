module vtl

// StridedTensorIterator is a struct to hold a Tensors
// iteration state while iterating through a Tensor
pub struct StridedTensorIterator {
	tensor       Tensor
pub mut:
	coord        &int
	backstrides  &int
	pos          int
	next_handler fn (mut s StridedTensorIterator) voidptr
}

// tensor_to_varray<T> returns the flatten representation of a tensor in a v array storing
// elements of type T
pub fn tensor_to_varray<T>(t Tensor) []T {
	mut arr := []T{}
	mut iter := t.iterator()
	for _ in 0 .. t.size {
		arr.push(iter.next())
	}
	return arr
}

// iterator creates an iterator through a Tensor
pub fn (t Tensor) iterator() StridedTensorIterator {
	if t.is_rowmajor_contiguous() {
		return t.rowmajor_contiguous_iterator()
	}
	return t.strided_iterator()
}

fn (t Tensor) rowmajor_contiguous_iterator() StridedTensorIterator {
	bs := 0
	return StridedTensorIterator{
		pos: 0
		next_handler: handle_flatten_iteration
		backstrides: &bs
		coord: &bs
		tensor: t
	}
}

fn (t Tensor) strided_iterator() StridedTensorIterator {
	coord := []int{len: t.rank()}
	return StridedTensorIterator{
		coord: &int(&coord[0])
		backstrides: tensor_backstrides(t)
		tensor: t
		pos: 0
		next_handler: handle_strided_iteration
	}
}

// handle_strided_iteration advances through a non-rowmajor-contiguous
// Tensor in Row-Major order
[unsafe]
fn handle_strided_iteration(mut s StridedTensorIterator) voidptr {
        // get current value after update new position
        val := storage_get(s.tensor.data, s.pos)

	rank := s.tensor.rank()
	shape := s.tensor.shape
	strides := s.tensor.strides
	unsafe {
		for i := rank - 1; i >= 0; i-- {
			if s.coord[i] < shape[i] - 1 {
				s.coord[i]++
				s.pos += strides[i]
				break
			} else {
				s.coord[i] = 0
				s.pos -= s.backstrides[i]
			}
		}
	}
        return val
}

// handle_flatten_iteration advances through a rowmajor-contiguous Tensor
// in Row-Major order
[inline]
fn handle_flatten_iteration(mut s StridedTensorIterator) voidptr {
        // get current value after update new position
        val := storage_get(s.tensor.data, s.pos)
        s.pos++
	return val
}

// next calls the iteration type for a given iterator
// which is either flat or strided and returns a voidptr containing the current value
[inline]
pub fn (mut s StridedTensorIterator) next() voidptr {
	return s.next_handler(s)
}

fn tensor_backstrides(t Tensor) &int {
	rank := t.rank()
	shape := t.shape
	strides := t.strides
	mut backstrides := []int{len: rank}
	for i := 0; i < rank; i++ {
		backstrides[i] = strides[i] * (shape[i] - 1)
	}
	return &int(backstrides.data)
}
