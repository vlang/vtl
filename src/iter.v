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
pub fn (mut s TensorIterator<T>) next<T>() ?(T, []int) {
	if s.iteration >= s.tensor.size() {
		return none
	}
	defer {
		s.iteration++
	}
	if s.next_handler == .flatten_iteration {
		return handle_flatten_iteration<T>(mut s), s.tensor.nth_index(s.iteration)
	}
	return handle_strided_iteration<T>(mut s), s.tensor.nth_index(s.iteration)
}

// iterators creates an array of iterators through a list of tensors
pub fn (t &Tensor<T>) iterators<T>(ts []&Tensor<T>) ?([]&TensorIterator<T>, []int) {
	mut next_ts := [t]
	for t_ in ts {
		next_ts << t_
	}
	if next_ts.len == 1 {
		return [t.iterator<T>()], t.shape
	}
	broadcasted_ts := broadcast_n<T>(next_ts)?
	shape := broadcasted_ts[0].shape
	mut iters := []&TensorIterator<T>{cap: broadcasted_ts.len}
	for t_ in broadcasted_ts {
		iters << t_.iterator<T>()
	}
	return iters, shape
}

// next calls the iteration type for a given list of iterators
// which is either flat or strided and returns a list of Nums containing the current values
[inline]
pub fn (mut its []&TensorIterator<T>) next<T>() ?([]T, []int) {
	return iterators_next<T>(mut its)
}

// iterators_next calls the iteration type for a given list of iterators
// which is either flat or strided and returns a list of Nums containing the current values
pub fn iterators_next<T>(mut its []&TensorIterator<T>) ?([]T, []int) {
	mut nums := []T{cap: its.len}
	mut index := []int{}
	for i, mut iter in its {
		val, index_ := iter.next() or { return err }
		if i == 0 {
			index = index_.clone()
		}
		nums << new_t<T>(val)
	}
	return nums, index
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
