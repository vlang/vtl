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

fn new_cpu<T>(len int, cap int, init T) &CpuStorage<T> {
	mut capacity := if cap < len { len } else { cap }
	capacity = imax(capacity, storage.vector_minimum_capacity)
	return &CpuStorage<T>{
		data: []T{len: len, cap: capacity, init: init}
	}
}

fn cpu_from_array<T>(arr []T) &CpuStorage<T> {
	return &CpuStorage<T>{
		data: arr.clone()
	}
}

// Private function. Used to implement CpuStorage operator
[inline]
fn (cpu &CpuStorage<T>) get<T>(i int) T {
	return cpu.data[i]
}

// Private function. Used to implement assigment to the CpuStorage element
[inline]
fn (mut cpu CpuStorage<T>) set<T>(i int, val T) {
	cpu.data[i] = val
}

// fill fills an entire storage with a given value
fn (mut cpu CpuStorage<T>) fill<T>(val T) {
	for i in 0 .. cpu.data.len {
		cpu.data[i] = val
	}
}

// clone returns an independent copy of a given CpuStorage
[inline]
fn (cpu &CpuStorage<T>) clone<T>() &CpuStorage<T> {
	return &CpuStorage<T>{
		data: cpu.data.clone()
	}
}

// like returns an independent copy of a given CpuStorage
[inline]
fn (cpu &CpuStorage<T>) like<T>() &CpuStorage<T> {
	return &CpuStorage<T>{
		data: []T{len: cpu.data.len, cap: cpu.data.cap}
	}
}

// like_with_len returns an independent copy of a given CpuStorage
[inline]
fn (cpu &CpuStorage<T>) like_with_len<T>(len int) &CpuStorage<T> {
	mut capacity := if cpu.data.cap < len { len } else { cpu.data.cap }
	return &CpuStorage<T>{
		data: []T{len: len, cap: capacity}
	}
}

fn (cpu &CpuStorage<T>) offset<T>(start int) &CpuStorage<T> {
	return &CpuStorage<T>{
		data: cpu.data[start..cpu.data.len]
	}
}

[inline]
fn (cpu &CpuStorage<T>) to_array<T>() []T {
	return cpu.data.clone()
}

[inline]
fn imax(a int, b int) int {
	return if a > b { a } else { b }
}
