module vtl

// as_bool casts the Tensor to a Tensor of bools.
[inline]
pub fn (t &Tensor[T]) as_bool[T]() &Tensor[bool] {
	// @todo: Implement using map
	mut iter := t.iterator[T]()
	mut ret := empty[bool](t.shape)
	for {
		val, i := iter.next() or { break }
		bool_val := td[T](val).bool()
		ret.set(i, bool_val)
	}
	return ret
}

// as_f64 casts the Tensor to a Tensor of f64s.
[inline]
pub fn (t &Tensor[T]) as_f64[T]() &Tensor[f64] {
	// @todo: Implement using map
	mut iter := t.iterator[T]()
	mut ret := empty[f64](t.shape)
	for {
		val, i := iter.next() or { break }
		f64_val := td[T](val).f64()
		ret.set(i, f64_val)
	}
	return ret
}

// as_int casts the Tensor to a Tensor of ints.
[inline]
pub fn (t &Tensor[T]) as_int[T]() &Tensor[int] {
	// @todo: Implement using map
	mut iter := t.iterator[T]()
	mut ret := empty[int](t.shape)
	for {
		val, i := iter.next() or { break }
		int_val := td[T](val).int()
		ret.set(i, int_val)
	}
	return ret
}
