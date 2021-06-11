module vtl

import vtl.etype
import vtl.storage

// IteratorHandler defines a function to use in order to mutate
// iteration position
pub type IteratorHandler = fn (mut s TensorIterator) etype.Num

// TensorIterator is a struct to hold a Tensors
// iteration state while iterating through a Tensor
[heap]
pub struct TensorIterator {
	tensor       Tensor
	next_handler IteratorHandler
mut:
	coord       &int
	backstrides &int
	iteration   int
	pos         int
}

// iterator creates an iterator through a Tensor
pub fn (t Tensor) iterator() TensorIterator {
	if t.is_rowmajor_contiguous() {
		return t.rowmajor_contiguous_iterator()
	}
	return t.strided_iterator()
}

fn (t Tensor) rowmajor_contiguous_iterator() TensorIterator {
	coord := 0
	bs := 0
	return t.custom_iterator(coord: &coord, backstrides: &bs, next_handler: handle_flatten_iteration)
}

fn (t Tensor) strided_iterator() TensorIterator {
	coord := []int{len: t.rank()}
	return t.custom_iterator(
		coord: unsafe { &int(&coord[0]) }
		backstrides: tensor_backstrides(t)
		next_handler: handle_strided_iteration
		pos: t.strided_offset()
	)
}

pub struct IteratorBuildData {
	next_handler IteratorHandler
	coord        &int
	backstrides  &int
	pos          int
}

// iterator creates an iterator through a Tensor with custom data
pub fn (t Tensor) custom_iterator(data IteratorBuildData) TensorIterator {
	return TensorIterator{
		coord: data.coord
		backstrides: data.backstrides
		tensor: t
		pos: data.pos
		next_handler: data.next_handler
	}
}

// handle_strided_iteration advances through a non-rowmajor-contiguous
// Tensor in Row-Major order
[unsafe]
fn handle_strided_iteration(mut s TensorIterator) etype.Num {
	// get current value after update new position
	val := storage.storage_get(s.tensor.data, s.pos, s.tensor.etype)
	rank := s.tensor.rank()
	shape := s.tensor.shape
	strides := s.tensor.strides

	unsafe {
		for k := rank - 1; k >= 0; k-- {
			if s.coord[k] < shape[k] - 1 {
				s.coord[k]++
				s.pos += strides[k]
				break
			} else {
				if k == 0 {
					// this will make the iterator finish
					s.iteration = s.tensor.size
				}
				s.coord[k] = 0
				s.pos -= s.backstrides[k]
			}
		}
	}
	return val
}

// handle_flatten_iteration advances through a rowmajor-contiguous Tensor
// in Row-Major order
[inline]
fn handle_flatten_iteration(mut s TensorIterator) etype.Num {
	// get current value after update new position
	val := storage.storage_get(s.tensor.data, s.pos, s.tensor.etype)
	s.pos++
	return val
}

// next calls the iteration type for a given iterator
// which is either flat or strided and returns a Num containing the current value
[inline]
pub fn (mut s TensorIterator) next() ?etype.Num {
	if s.iteration >= s.tensor.size {
		return none
	}
	s.iteration++
	val := s.next_handler(mut s)
	return val
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

// Iterate with n tensors
// iterators creates an array of iterators through a list of tensors
pub fn (t Tensor) iterators(ts ...Tensor) []TensorIterator {
	mut iters := []TensorIterator{cap: ts.len + 1}
	iters << t.iterator()
	for i := 0; i < ts.len; i++ {
		tib := ts[i].broadcast_to(t.shape)
		iters << tib.iterator()
	}
	return iters
}

// Iterate with n tensors
// iterators creates an array of iterators through a list of tensors
pub fn (ts []Tensor) iterators() []TensorIterator {
	if ts.len == 0 {
		return []TensorIterator{}
	}
	mut iters := []TensorIterator{cap: ts.len}
	for i := 0; i < ts.len; i++ {
		tib := ts[i].broadcast_to(ts[0].shape)
		iters << tib.iterator()
	}
	return iters
}

// next calls the iteration type for a given list of iterators
// which is either flat or strided and returns a list of Nums containing the current values
[inline]
pub fn (mut its []TensorIterator) next() []etype.Num {
	mut nums := []etype.Num{cap: its.len}
	for mut iter in its {
		if val := iter.next() {
			nums << val
		}
	}
	return nums
}
