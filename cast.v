module vtl

fn cast_tensor_values[T, U](t &Tensor[T]) &Tensor[U] {
	mut result := empty[U](t.shape, memory: .row_major)
	for flat_index in 0 .. t.size {
		result.data.data[flat_index] = cast[U](td[T](t.get_nth[T](flat_index)))
	}
	return result
}

// as_bool casts the Tensor to a Tensor of bools.
// If the original Tensor is not a Tensor of bools, then each value is cast to a bool,
// otherwise the original Tensor is returned.

// as_bool exposes this operation as part of the public API.

// as_bool exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) as_bool[T]() &Tensor[bool] {
	$if T is bool {
		return t
	} $else {
		return cast_tensor_values[T, bool](t)
	}
}

// as_f32 casts the Tensor to a Tensor of f32s.
// If the original Tensor is not a Tensor of f32s, then each value is cast to a f32,
// otherwise the original Tensor is returned.

// as_f32 exposes this operation as part of the public API.

// as_f32 exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) as_f32[T]() &Tensor[f32] {
	$if T is f32 {
		return t
	} $else {
		return cast_tensor_values[T, f32](t)
	}
}

// as_f64 casts the Tensor to a Tensor of f64s.
// If the original Tensor is not a Tensor of f64s, then each value is cast to a f64,
// otherwise the original Tensor is returned.

// as_f64 exposes this operation as part of the public API.

// as_f64 exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) as_f64[T]() &Tensor[f64] {
	$if T is f64 {
		return t
	} $else {
		return cast_tensor_values[T, f64](t)
	}
}

// as_i64 casts each tensor value to i64 while preserving its logical shape.
pub fn (t &Tensor[T]) as_i64[T]() &Tensor[i64] {
	$if T is i64 {
		return t
	} $else {
		return cast_tensor_values[T, i64](t)
	}
}

// as_i32 casts each tensor value to i32 while preserving its logical shape.
pub fn (t &Tensor[T]) as_i32[T]() &Tensor[i32] {
	$if T is i32 {
		return t
	} $else {
		return cast_tensor_values[T, i32](t)
	}
}

// as_i16 casts tensor values to signed 16-bit integers.

// as_i16 exposes this operation as part of the public API.

// as_i16 exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) as_i16[T]() &Tensor[i16] {
	$if T is i16 {
		return t
	} $else {
		return cast_tensor_values[T, i16](t)
	}
}

// as_i8 casts tensor values to signed 8-bit integers.

// as_i8 exposes this operation as part of the public API.

// as_i8 exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) as_i8[T]() &Tensor[i8] {
	$if T is i8 {
		return t
	} $else {
		return cast_tensor_values[T, i8](t)
	}
}

// as_int casts tensor values to V ints.

// as_int exposes this operation as part of the public API.

// as_int exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) as_int[T]() &Tensor[int] {
	$if T is int {
		return t
	} $else {
		return cast_tensor_values[T, int](t)
	}
}

// as_string casts the Tensor to a Tensor of string values.
// If the original Tensor is not a Tensor of strings, then each value is cast to a string,
// otherwise the original Tensor is returned.

// as_string exposes this operation as part of the public API.

// as_string exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) as_string[T]() &Tensor[string] {
	$if T is string {
		return t
	} $else {
		return cast_tensor_values[T, string](t)
	}
}

// as_u8 casts tensor values to unsigned 8-bit integers.

// as_u8 exposes this operation as part of the public API.

// as_u8 exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) as_u8[T]() &Tensor[u8] {
	$if T is u8 {
		return t
	} $else {
		return cast_tensor_values[T, u8](t)
	}
}

// as_u16 casts each tensor value to u16 while preserving its logical shape.
pub fn (t &Tensor[T]) as_u16[T]() &Tensor[u16] {
	$if T is u16 {
		return t
	} $else {
		return cast_tensor_values[T, u16](t)
	}
}

// as_u32 casts each tensor value to u32 while preserving its logical shape.
pub fn (t &Tensor[T]) as_u32[T]() &Tensor[u32] {
	$if T is u32 {
		return t
	} $else {
		return cast_tensor_values[T, u32](t)
	}
}

// as_u64 casts each tensor value to u64 while preserving its logical shape.
pub fn (t &Tensor[T]) as_u64[T]() &Tensor[u64] {
	$if T is u64 {
		return t
	} $else {
		return cast_tensor_values[T, u64](t)
	}
}
