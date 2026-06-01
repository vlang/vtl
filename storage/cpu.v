module storage

// vector_minimum_capacity is a public constant used by this module.
pub const vector_minimum_capacity = 2
// vector_growth_factor is a public constant used by this module.
pub const vector_growth_factor = 2
// vector_shrink_threshold is a public constant used by this module.
pub const vector_shrink_threshold = 1.0 / 4.0

// CpuStorage

// CpuStorage defines a public data structure for this module.

// CpuStorage defines a public data structure for this module.
@[heap]
pub struct CpuStorage[T] {
pub mut:
	data []T
}

// storage exposes this operation as part of the public API.
pub fn storage[T](len int, cap int, init T) &CpuStorage[T] {
	mut capacity := if cap < len { len } else { cap }
	capacity = imax(capacity, vector_minimum_capacity)
	return &CpuStorage[T]{
		data: []T{len: len, cap: capacity, init: init}
	}
}

// from_array exposes this operation as part of the public API.
pub fn from_array[T](arr []T) &CpuStorage[T] {
	return &CpuStorage[T]{
		data: arr.clone()
	}
}

// Private function. Used to implement the Storage operator

// get exposes this operation as part of the public API.

// get exposes this operation as part of the public API.
@[inline]
pub fn (s &CpuStorage[T]) get[T](i int) T {
	return s.data[i]
}

// Private function. Used to implement assignments to the Storage element

// set exposes this operation as part of the public API.

// set exposes this operation as part of the public API.
@[inline]
pub fn (mut s CpuStorage[T]) set[T](i int, val T) {
	s.data[i] = val
}

// fill fills an entire storage with a given value
pub fn (mut s CpuStorage[T]) fill[T](val T) {
	for i in 0 .. s.data.len {
		s.data[i] = val
	}
}

// clone returns an independent copy of a given Storage

// clone exposes this operation as part of the public API.

// clone exposes this operation as part of the public API.
@[inline]
pub fn (s &CpuStorage[T]) clone[T]() &CpuStorage[T] {
	return &CpuStorage[T]{
		data: s.data.clone()
	}
}

// like returns an independent copy of a given Storage

// like exposes this operation as part of the public API.

// like exposes this operation as part of the public API.
@[inline]
pub fn (s &CpuStorage[T]) like[T]() &CpuStorage[T] {
	return &CpuStorage[T]{
		data: []T{len: s.data.len, cap: s.data.cap}
	}
}

// like_with_len returns an independent copy of a given Storage

// like_with_len exposes this operation as part of the public API.

// like_with_len exposes this operation as part of the public API.
@[inline]
pub fn (s &CpuStorage[T]) like_with_len[T](len int) &CpuStorage[T] {
	mut capacity := if s.data.cap < len { len } else { s.data.cap }
	return &CpuStorage[T]{
		data: []T{len: len, cap: capacity}
	}
}

// offset exposes this operation as part of the public API.
pub fn (s &CpuStorage[T]) offset[T](start int) &CpuStorage[T] {
	return &CpuStorage[T]{
		data: s.data[start..s.data.len]
	}
}

// to_array exposes this operation as part of the public API.

// to_array exposes this operation as part of the public API.
@[inline]
pub fn (s &CpuStorage[T]) to_array[T]() []T {
	return s.data.clone()
}

@[inline]
fn imax(a int, b int) int {
	return if a > b { a } else { b }
}
