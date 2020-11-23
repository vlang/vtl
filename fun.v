module vtl

pub type MapFn = fn(x voidptr, i int) voidptr

pub type ApplyFn = fn(x voidptr, i int) voidptr

// map maps a function to a given Tensor retuning a new Tensor with same shape
pub fn (t Tensor) map(f MapFn) Tensor {
        mut ret := new_tensor_like(t)
        mut iter := t.iterator()
	for i in 0 .. t.size {
		unsafe {
                        val := f(iter.next(), i)
                        storage_set(ret.data, i, &val)
                }
	}
	return ret
}

// apply applies a function to each element of a given Tensor
pub fn (t Tensor) apply(f ApplyFn) {
        mut iter := t.iterator()
	for i in 0 .. t.size {
		unsafe {
                        val := f(iter.next(), i)
                        storage_set(t.data, i, &val)
                }
	}
}
