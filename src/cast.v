module vtl

// as_bool casts the Tensor to a Tensor of bools.
// If the original Tensor is not a Tensor of bools, then each value is cast to a bool,
// otherwise the original Tensor is returned.
[inline]
pub fn (t &Tensor[T]) as_bool[T]() &Tensor[bool] {
	$if T is bool {
		return t
	} $else {
		// TODO: Implement using map
		mut iter := t.iterator[T]()
		mut ret := empty[bool](t.shape)
		for {
			val, i := iter.next() or { break }
			bool_val := td[T](val).bool()
			ret.set(i, bool_val)
		}
		return ret
	}
}

// as_f32 casts the Tensor to a Tensor of f32s.
// If the original Tensor is not a Tensor of f32s, then each value is cast to a f32,
// otherwise the original Tensor is returned.
[inline]
pub fn (t &Tensor[T]) as_f32[T]() &Tensor[f32] {
	$if T is f32 {
		return t
	} $else {
		// TODO: Implement using map
		mut iter := t.iterator[T]()
		mut ret := empty[f32](t.shape)
		for {
			val, i := iter.next() or { break }
			f32_val := td[T](val).f32()
			ret.set(i, f32_val)
		}
		return ret
	}
}

// as_f64 casts the Tensor to a Tensor of f64s.
// If the original Tensor is not a Tensor of f64s, then each value is cast to a f64,
// otherwise the original Tensor is returned.
[inline]
pub fn (t &Tensor[T]) as_f64[T]() &Tensor[f64] {
	$if T is f64 {
		return t
	} $else {
		// TODO: Implement using map
		mut iter := t.iterator[T]()
		mut ret := empty[f64](t.shape)
		for {
			val, i := iter.next() or { break }
			f64_val := td[T](val).f64()
			ret.set(i, f64_val)
		}
		return ret
	}
}

// as_i16 casts the Tensor to a Tensor of i16 values.
// If the original Tensor is not a Tensor of i16s, then each value is cast to a i16,
// otherwise the original Tensor is returned.
[inline]
pub fn (t &Tensor[T]) as_i16[T]() &Tensor[i16] {
	$if T is i16 {
		return t
	} $else {
		// TODO: Implement using map
		mut iter := t.iterator[T]()
		mut ret := empty[i16](t.shape)
		for {
			val, i := iter.next() or { break }
			i16_val := td[T](val).i16()
			ret.set(i, i16_val)
		}
		return ret
	}
}

// as_i8 casts the Tensor to a Tensor of i8 values.
// If the original Tensor is not a Tensor of i8s, then each value is cast to a i8,
// otherwise the original Tensor is returned.
[inline]
pub fn (t &Tensor[T]) as_i8[T]() &Tensor[i8] {
	$if T is i8 {
		return t
	} $else {
		// TODO: Implement using map
		mut iter := t.iterator[T]()
		mut ret := empty[i8](t.shape)
		for {
			val, i := iter.next() or { break }
			i8_val := td[T](val).i8()
			ret.set(i, i8_val)
		}
		return ret
	}
}

// as_int casts the Tensor to a Tensor of ints.
// If the original Tensor is not a Tensor of ints, then each value is cast to a int,
// otherwise the original Tensor is returned.
[inline]
pub fn (t &Tensor[T]) as_int[T]() &Tensor[int] {
	$if T is int {
		return t
	} $else {
		// TODO: Implement using map
		mut iter := t.iterator[T]()
		mut ret := empty[int](t.shape)
		for {
			val, i := iter.next() or { break }
			int_val := td[T](val).int()
			ret.set(i, int_val)
		}
		return ret
	}
}

// as_string casts the Tensor to a Tensor of string values.
// If the original Tensor is not a Tensor of strings, then each value is cast to a string,
// otherwise the original Tensor is returned.
[inline]
pub fn (t &Tensor[T]) as_string[T]() &Tensor[string] {
	$if T is string {
		return t
	} $else {
		// TODO: Implement using map
		mut iter := t.iterator[T]()
		mut ret := empty[string](t.shape)
		for {
			val, i := iter.next() or { break }
			string_val := td[T](val).string()
			ret.set(i, string_val)
		}
		return ret
	}
}

// as_u8 casts the Tensor to a Tensor of u8 values.
// If the original Tensor is not a Tensor of u8s, then each value is cast to a u8,
// otherwise the original Tensor is returned.
[inline]
pub fn (t &Tensor[T]) as_u8[T]() &Tensor[u8] {
	$if T is u8 {
		return t
	} $else {
		// TODO: Implement using map
		mut iter := t.iterator[T]()
		mut ret := empty[u8](t.shape)
		for {
			val, i := iter.next() or { break }
			u8_val := td[T](val).u8()
			ret.set(i, u8_val)
		}
		return ret
	}
}
