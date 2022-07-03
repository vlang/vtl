module vtl

// IteratorStrategy defines a function to use in order to mutate
// iteration position
pub enum IteratorStrategy {
	flatten_iteration
	strided_iteration
}

// TensorIterator is a struct to hold a Tensors
// iteration state while iterating through a Tensor
[heap]
pub struct TensorIterator<T> {
pub:
	tensor       &Tensor<T>
	next_handler IteratorStrategy
pub mut:
	iteration int
}

// iterator creates an iterator through a Tensor
pub fn (t &Tensor<T>) iterator<T>() &TensorIterator<T> {
	if t.is_row_major_contiguous() {
		return t.row_major_contiguous_iterator()
	}
	return t.strided_iterator()
}

pub struct IteratorBuildData<T> {
	next_handler IteratorStrategy
	start        int
}

// iterator creates an iterator through a Tensor with custom data
pub fn (t &Tensor<T>) custom_iterator<T>(data IteratorBuildData<T>) &TensorIterator<T> {
	return &TensorIterator<T>{
		tensor: t
		iteration: data.start
		next_handler: data.next_handler
	}
}

// next calls the iteration type for a given iterator
// which is either flat or strided and returns a Num containing the current value
[inline]
pub fn (mut s TensorIterator<T>) next<T>() ?(T, int) {
	if s.iteration >= s.tensor.size() {
		return none
	}
	defer {
		s.iteration++
	}
	if s.next_handler == .flatten_iteration {
		return handle_flatten_iteration<T>(mut s), s.iteration
	}
	return handle_strided_iteration<T>(mut s), s.iteration
}

// iterators creates an array of iterators through a list of tensors
pub fn (t &Tensor<T>) iterators<T>(ts []&Tensor<T>) []&TensorIterator<T> {
	mut next_ts := [t]
	for t_ in ts {
		next_ts << t_
	}
	return iterators<T>(next_ts)
}

// iterators creates an array of iterators through a list of tensors
[inline]
pub fn (ts []&Tensor<T>) iterators<T>() []&TensorIterator<T> {
	return iterators<T>(ts)
}

// iterators creates an array of iterators through a list of tensors
pub fn iterators<T>(ts []&Tensor<T>) []&TensorIterator<T> {
	if ts.len == 0 {
		return []&TensorIterator<T>{}
	}
	mut iters := []&TensorIterator<T>{cap: ts.len}
	for i in 0 .. ts.len {
		tib := ts[i].broadcast_to<T>(ts[0].shape)
		iters << tib.iterator<T>()
	}
	return iters
}

// next calls the iteration type for a given list of iterators
// which is either flat or strided and returns a list of Nums containing the current values
[inline]
pub fn (mut its []&TensorIterator<T>) next<T>() ?([]T, int) {
	return iterators_next<T>(mut its)
}

// iterators_next calls the iteration type for a given list of iterators
// which is either flat or strided and returns a list of Nums containing the current values
pub fn iterators_next<T>(mut its []&TensorIterator<T>) ?([]T, int) {
	mut nums := []T{cap: its.len}
	mut n := -1
	for i, mut iter in its {
		val, n_ := iter.next() or { return err }
		if i == 0 {
			n = n_
		}
		nums << T(val)
	}
	return nums, n
}

fn (t &Tensor<T>) row_major_contiguous_iterator<T>() &TensorIterator<T> {
	return t.custom_iterator<T>(
		next_handler: .flatten_iteration
	)
}

fn (t &Tensor<T>) strided_iterator<T>() &TensorIterator<T> {
	return t.custom_iterator<T>(
		next_handler: .strided_iteration
	)
}

// handle_strided_iteration advances through a non-row_major-contiguous
// Tensor in Row-Major order
fn handle_strided_iteration<T>(mut s TensorIterator<T>) T {
	// get current value after update new position
	val := s.tensor.get_nth(s.iteration)
	return val
}

// handle_flatten_iteration advances through a row_major-contiguous Tensor
// in Row-Major order
fn handle_flatten_iteration<T>(mut s TensorIterator<T>) T {
	// get current value after update new position
	val := s.tensor.get_nth(s.iteration)
	return val
}

fn tensor_backstrides<T>(t &Tensor<T>) []int {
	rank := t.rank()
	shape := t.shape
	strides := t.strides
	mut backstrides := []int{len: rank}
	for i := 0; i < rank; i++ {
		backstrides[i] = strides[i] * (shape[i] - 1)
	}
	return backstrides
}
