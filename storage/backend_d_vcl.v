module storage

pub const (
	vector_minimum_capacity = 2
	vector_growth_factor    = 2
	vector_shrink_threshold = 1.0 / 4.0
)

// Storage
[heap]
pub struct Storage<T> {
pub mut:
	data []T
}

pub fn new_storage<T>(len int, cap int, init T) &Storage<T> {
	todo(@MOD + '.' + @FN)
	mut capacity := if cap < len { len } else { cap }
	capacity = imax(capacity, storage.vector_minimum_capacity)
	return &Storage<T>{
		data: []T{len: len, cap: capacity, init: init}
	}
}

pub fn from_array<T>(arr []T) &Storage<T> {
	todo(@MOD + '.' + @FN)
	return &Storage<T>{
		data: arr.clone()
	}
}

// Private function. Used to implement Storage operator
[inline]
pub fn (storage &Storage<T>) get<T>(i int) T {
	return storage.data[i]
}

// Private function. Used to implement assigment to the Storage element
[inline]
pub fn (mut storage Storage<T>) set<T>(i int, val T) {
	storage.data[i] = val
}

// fill fills an entire storage with a given value
pub fn (mut storage Storage<T>) fill<T>(val T) {
	for i in 0 .. storage.data.len {
		storage.data[i] = val
	}
}

// clone returns an independent copy of a given Storage
[inline]
pub fn (storage &Storage<T>) clone<T>() &Storage<T> {
	return &Storage<T>{
		data: storage.data.clone()
	}
}

// like returns an independent copy of a given Storage
[inline]
pub fn (storage &Storage<T>) like<T>() &Storage<T> {
	return &Storage<T>{
		data: []T{len: storage.data.len, cap: storage.data.cap}
	}
}

// like_with_len returns an independent copy of a given Storage
[inline]
pub fn (storage &Storage<T>) like_with_len<T>(len int) &Storage<T> {
	mut capacity := if storage.data.cap < len { len } else { storage.data.cap }
	return &Storage<T>{
		data: []T{len: len, cap: capacity}
	}
}

pub fn (storage &Storage<T>) offset<T>(start int) &Storage<T> {
	return &Storage<T>{
		data: storage.data[start..storage.data.len]
	}
}

[inline]
pub fn (storage &Storage<T>) to_array<T>() []T {
	return storage.data.clone()
}

[inline]
fn imax(a int, b int) int {
	return if a > b { a } else { b }
}

[inline]
fn todo(func_name string) {
	$if debug ? {
		eprintln('$func_name: This function is not implemented yet for VCL backend. Using CPU storage for now.')
	}
}
