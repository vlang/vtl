module vtl

// StridedIterator is a struct to hold a Tensors
// iteration state while iterating through a Tensor
pub struct StridedIterator {
	rank         int
	shape        []int
	strides      []int
pub mut:
	coord        &int
	backstrides  &int
	pos          int
	next_handler fn (mut s StridedIterator)
}

// tensor_to_varray<T> returns the flatten representation of a tensor in a v array storing
// elements of type T
pub fn tensor_to_varray<T>(t Tensor) []T {
	mut arr := []T{}
	mut iter := t.iterator()
	for _ in 0 .. t.size {
		arr.push(storage_get(t.data, iter.pos))
		iter.next()
	}
	return arr
}

// iterator creates an iterator through a Tensor
pub fn (t Tensor) iterator() StridedIterator {
	if t.is_rowmajor_contiguous() {
		return t.rowmajor_contiguous_iterator()
	}
	return t.strided_iterator()
}

fn (t Tensor) rowmajor_contiguous_iterator() StridedIterator {
        bs := 0
        return StridedIterator{
                pos: 0
                next_handler: handle_flatten_iteration
                backstrides: &bs
                coord: &bs
        }
}

fn (t Tensor) strided_iterator() StridedIterator {
        coord := []int{len: t.rank()}

	return StridedIterator{
		coord: &int(&coord[0])
		backstrides: tensor_backstrides(t)
		rank: t.rank()
		shape: t.shape
		strides: t.strides
		pos: 0
		next_handler: handle_strided_iteration
	}
}

// handle_strided_iteration advances through a non-rowmajor-contiguous
// Tensor in Row-Major order
fn handle_strided_iteration(mut s StridedIterator) {
	unsafe {
		for i := s.rank - 1; i >= 0; i-- {
			if s.coord[i] < s.shape[i] - 1 {
				s.coord[i]++
				s.pos += s.strides[i]
				break
			} else {
				s.coord[i] = 0
				s.pos -= s.backstrides[i]
			}
		}
	}
}

fn tensor_backstrides(t Tensor) &int {
        mut backstrides := []int{len: t.rank()}
	for i := 0; i < t.rank(); i++ {
		backstrides[i] = t.strides[i] * (t.shape[i] - 1)
	}
        return &int(backstrides.data)
}

// handle_flatten_iteration advances through a rowmajor-contiguous Tensor
// in Row-Major order
[inline]
fn handle_flatten_iteration(mut s StridedIterator) {
	s.pos++
}

// next calls the iteration type for a given iterator
// which is either flat or strided and returns a voidptr containing the current value
[inline]
pub fn (mut s StridedIterator) next() {
	s.next_handler(s)
}
