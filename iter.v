module vtl

import vtl.storage

// IteratorHandler defines a function to use in order to mutate
// iteration position
pub enum IteratorHandler {
	handle_flatten_iteration
	handle_strided_iteration
}

// TensorIterator is a struct to hold a Tensors
// iteration state while iterating through a Tensor
[heap]
pub struct TensorIterator<T> {
pub:
	tensor       &Tensor<T>
	next_handler IteratorHandler
pub mut:
	coord       &int
	backstrides &int
	iteration   int
	pos         int
}

// iterator creates an iterator through a Tensor
pub fn (t &Tensor<T>) iterator<T>() &TensorIterator<T> {
	if t.is_rowmajor_contiguous() {
		return t.rowmajor_contiguous_iterator()
	}
	return t.strided_iterator()
}

fn (t &Tensor<T>) rowmajor_contiguous_iterator<T>() &TensorIterator<T> {
	coord := 0
	bs := 0
	return t.custom_iterator<T>(
		coord: &coord
		backstrides: &bs
		next_handler: .handle_flatten_iteration
	)
}

fn (t &Tensor<T>) strided_iterator<T>() &TensorIterator<T> {
	coord := []int{len: t.rank()}
	return t.custom_iterator<T>(
		coord: unsafe { &int(&coord[0]) }
		backstrides: tensor_backstrides<T>(t)
		next_handler: .handle_strided_iteration
		pos: t.strided_offset_index()
	)
}

pub struct IteratorBuildData<T> {
	next_handler IteratorHandler
	coord        &int
	backstrides  &int
	pos          int
}

// iterator creates an iterator through a Tensor with custom data
pub fn (t &Tensor<T>) custom_iterator<T>(data IteratorBuildData<T>) &TensorIterator<T> {
	return &TensorIterator<T>{
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
fn handle_strided_iteration<T>(mut s TensorIterator<T>) T {
	// get current value after update new position
	val := storage.storage_get<T>(s.tensor.data, s.pos)
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
pub fn handle_flatten_iteration<T>(mut s TensorIterator<T>) T {
	// get current value after update new position
	val := storage.storage_get<T>(s.tensor.data, s.pos)
	defer {
		s.pos++
	}
	return val
}

// next calls the iteration type for a given iterator
// which is either flat or strided and returns a Num containing the current value
[inline]
pub fn (mut s TensorIterator<T>) next<T>() ?T {
	if s.iteration >= s.tensor.size {
		return none
	}
	defer {
		s.iteration++
	}
	return if s.next_handler == .handle_flatten_iteration {
		handle_flatten_iteration<T>(mut s)
	} else {
		unsafe { handle_strided_iteration<T>(mut s) }
	}
}

fn tensor_backstrides<T>(t &Tensor<T>) &int {
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
pub fn (t &Tensor<T>) iterators<T>(ts ...&&Tensor<T>) []&TensorIterator<T> {
	mut iters := []&TensorIterator<T>{cap: ts.len + 1}
	iters << t.iterator()
	for i in 0 .. ts.len {
		tib := ts[i].broadcast_to(t.shape)
		iters << tib.iterator()
	}
	return iters
}

// Iterate with n tensors
// iterators creates an array of iterators through a list of tensors
pub fn (ts []&Tensor<T>) iterators<T>() []&TensorIterator<T> {
	if ts.len == 0 {
		return []&TensorIterator<T>{}
	}
	mut iters := []&TensorIterator<T>{cap: ts.len}
	for i in 0 .. ts.len {
		tib := ts[i].broadcast_to(ts[0].shape)
		iters << tib.iterator()
	}
	return iters
}

// next calls the iteration type for a given list of iterators
// which is either flat or strided and returns a list of Nums containing the current values
[inline]
pub fn (mut its []&TensorIterator<T>) next<T>() ?[]T {
	mut nums := []T{cap: its.len}
	for mut iter in its {
		val := iter.next() ?
		nums << val
	}
	return nums
}
