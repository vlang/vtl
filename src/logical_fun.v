module vtl

// array_equal returns true if input arrays have the same shape and all elements equal.
pub fn (t &Tensor<T>) array_equal<T>(other &Tensor<T>) bool {
	if t.shape != other.shape {
		return false
	}
	mut iters, _ := t.iterators<T>([other]) or { return false }
	for {
		vals, _ := iterators_next<T>(mut iters) or { break }
		if vals[0] != vals[1] {
			return false
		}
	}
	return true
}

// array_equiv returns true if input arrays are shape consistent and all elements equal.
// Shape consistent means they are either the same shape,
// or one input array can be broadcasted to create the same shape as the other one.
pub fn (t &Tensor<T>) array_equiv<T>(other &Tensor<T>) bool {
	mut iters, _ := t.iterators<T>([other]) or { return false }
	for {
		vals, _ := iterators_next<T>(mut iters) or { break }
		if vals[0] != vals[1] {
			return false
		}
	}
	return true
}