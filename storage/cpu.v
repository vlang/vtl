module storage

pub const (
	vector_minimum_capacity = 2
	vector_growth_factor    = 2
	vector_shrink_threshold = 1.0 / 4.0
)

// CpuStorage - this implementation will change once Generics are working correctly
pub struct CpuStorage {
pub:
	element_size int
pub mut:
	data         voidptr
	len          int
	capacity     int
}

pub fn new_cpu(len int, capacity int, element_size int) CpuStorage {
	mut capacity_ := if capacity < len { len } else { capacity }
	capacity_ = max(capacity_, vector_minimum_capacity)
	return CpuStorage{
		len: len
		capacity: capacity_
		data: vcalloc(capacity_ * element_size)
		element_size: element_size
	}
}

pub fn new_cpu_with_default(len int, capacity int, element_size int, val voidptr) CpuStorage {
	mut capacity_ := if capacity < len { len } else { capacity }
	capacity_ = max(capacity_, vector_minimum_capacity)
	mut cpu := CpuStorage{
		len: len
		capacity: capacity_
		element_size: element_size
		data: vcalloc(capacity_ * element_size)
	}
	if val != 0 {
		for i in 0 .. cpu.len {
			unsafe {cpu.set_unsafe(i, val)}
		}
	}
	return cpu
}

pub fn new_array_from_c_array(len int, capacity int, element_size int, c_array voidptr) CpuStorage {
	capacity_ := if capacity < len { len } else { capacity }
	cpu := CpuStorage{
		element_size: element_size
		data: vcalloc(capacity_ * element_size)
		len: len
		capacity: capacity_
	}
	// TODO Write all memory functions (like memcpy) in V
	unsafe {C.memcpy(cpu.data, c_array, len * element_size)}
	return cpu
}

// Private function. Used to implement CpuStorage operator
pub fn (a CpuStorage) get(i int) voidptr {
	$if !no_bounds_checking ? {
		if i < 0 || i >= a.len {
			panic('CpuStorage.get: index out of range (i == $i, a.len == $a.len)')
		}
	}
	return unsafe {a.get_unsafe(i)}
}

// Private function. Used to implement assigment to the CpuStorage element.
pub fn (mut a CpuStorage) set(i int, val voidptr) {
	$if !no_bounds_checking ? {
		if i < 0 || i >= a.len {
			panic('CpuStorage.set: index out of range (i == $i, a.len == $a.len)')
		}
	}
	unsafe {a.set_unsafe(i, val)}
}

// we manually inline this for single operations for performance without -prod
[inline]
[unsafe]
fn (a CpuStorage) get_unsafe(i int) voidptr {
	unsafe {
		return byteptr(a.data) + i * a.element_size
	}
}

// we manually inline this for single operations for performance without -prod
[inline]
[unsafe]
fn (mut a CpuStorage) set_unsafe(i int, val voidptr) {
	unsafe {C.memcpy(byteptr(a.data) + a.element_size * i, val, a.element_size)}
}

// Apply growth factor if needed
[inline]
fn (mut a CpuStorage) ensure_capacity(required int) {
	if required <= a.capacity {
		return
	}
	mut capacity := if a.capacity < vector_minimum_capacity { vector_minimum_capacity } else { a.capacity *
			vector_growth_factor }
	for required > capacity {
		capacity *= vector_growth_factor
	}
	if a.capacity == vector_minimum_capacity {
		a.data = vcalloc(capacity * a.element_size)
	} else {
		a.data = v_realloc(a.data, u32(capacity * a.element_size))
	}
	a.capacity = capacity
}

[inline]
fn max(a int, b int) int {
	return if a > b {
		a
	} else {
		b
	}
}
