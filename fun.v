module vtl

pub type MapFn<T, U> = fn (x T, i int) U

pub type ApplyFn<T> = fn (x T, i int) T

pub type NMapFn<T, U> = fn (x []T, i int) U

pub type NApplyFn<T> = fn (x []T, i int)

// map maps a function to a given Tensor retuning a new Tensor with same shape
pub fn (t &Tensor<T>) map<T, U>(f MapFn<T, U>) &Tensor<U> {
	mut ret := new_tensor_like<T>(t).as_type<U>()
	mut iter := t.iterator()
	mut pos := iter.pos
	for _ in 0 .. ret.size() {
		if val := iter.next() {
			next_val := f(val, pos)
			storage.storage_set<U>(ret.data, pos, next_val)
			pos = iter.pos
		}
	}
	return ret
}

// map maps a function to a given list of Tensor retuning a new Tensor with same shape
pub fn (t &Tensor<T>) nmap<T, U>(f NMapFn<T, U>, ts ...Tensor<T>) &Tensor<U> {
	mut ret := new_tensor_like<T>(t).as_type<U>()
	mut iters := t.iterators(ts)
	for i in 0 .. ret.size() {
		if vals := iterators_next<T>(mut iters) {
			val := f(vals, i)
			storage.storage_set<U>(ret.data, i, val)
		}
	}
	return ret
}

// napply applies a function to each element of a given Tensor
pub fn (t &Tensor<T>) napply<T>(f NApplyFn<T>, ts ...Tensor<T>) {
	mut iters := t.iterators(ts)
	for i in 0 .. t.size() {
		if vals := iterators_next<T>(mut iters) {
			val := f(vals, i)
			storage.storage_set<T>(t.data, i, val)
		}
	}
}

// equal checks if two Tensors are equal
pub fn (t &Tensor<T>) equal<T>(other &Tensor<T>) bool {
	if t.shape != other.shape {
		return false
	}
	mut iters := t.iterators([other])
	for _ in 0 .. t.size() {
		if vals := iterators_next<T>(mut iters) {
			if vals[0] != vals[1] {
				return false
			}
		}
	}
	return true
}
