module storage

pub const (
	vector_minimum_capacity = 2
	vector_growth_factor    = 2
	vector_shrink_threshold = 1.0 / 4.0
)

// CpuStorage
[heap]
pub struct CpuStorage<T> {
pub mut:
	data []T
}

pub fn storage<T>(len int, cap int, init T) &CpuStorage<T> {
	mut capacity := if cap < len { len } else { cap }
	capacity = imax(capacity, storage.vector_minimum_capacity)
	return &CpuStorage<T>{
		data: []T{len: len, cap: capacity, init: init}
	}
}

pub fn from_array<T>(arr []T) &CpuStorage<T> {
	return &CpuStorage<T>{
		data: arr.clone()
	}
}

// Private function. Used to implement Storage operator
[inline]
pub fn (storage &CpuStorage<T>) get<T>(i int) T {
	return storage.data[i]
}

// Private function. Used to implement assigment to the Storage element
[inline]
pub fn (mut storage CpuStorage<T>) set<T>(i int, val T) {
	storage.data[i] = val
}

// fill fills an entire storage with a given value
pub fn (mut storage CpuStorage<T>) fill<T>(val T) {
	for i in 0 .. storage.data.len {
		storage.data[i] = val
	}
}

// clone returns an independent copy of a given Storage
[inline]
pub fn (storage &CpuStorage<T>) clone<T>() &CpuStorage<T> {
	return &CpuStorage<T>{
		data: storage.data.clone()
	}
}

// like returns an independent copy of a given Storage
[inline]
pub fn (storage &CpuStorage<T>) like<T>() &CpuStorage<T> {
	return &CpuStorage<T>{
		data: []T{len: storage.data.len, cap: storage.data.cap}
	}
}

// like_with_len returns an independent copy of a given Storage
[inline]
pub fn (storage &CpuStorage<T>) like_with_len<T>(len int) &CpuStorage<T> {
	mut capacity := if storage.data.cap < len { len } else { storage.data.cap }
	return &CpuStorage<T>{
		data: []T{len: len, cap: capacity}
	}
}

pub fn (storage &CpuStorage<T>) offset<T>(start int) &CpuStorage<T> {
	return &CpuStorage<T>{
		data: storage.data[start..storage.data.len]
	}
}

[inline]
pub fn (storage &CpuStorage<T>) to_array<T>() []T {
	return storage.data.clone()
}

[inline]
fn imax(a int, b int) int {
	return if a > b { a } else { b }
}
