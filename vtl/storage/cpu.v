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
	data     voidptr
	len      int
	capacity int
}

fn new_cpu(len int, capacity int, element_size int) Storage {
	mut capacity_ := if capacity < len { len } else { capacity }
	capacity_ = max(capacity_, storage.vector_minimum_capacity)
	return CpuStorage{
		len: len
		capacity: capacity_
		data: vcalloc(capacity_ * element_size)
		element_size: element_size
	}
}

fn new_cpu_with_default(len int, capacity int, element_size int, val voidptr) CpuStorage {
	mut capacity_ := if capacity < len { len } else { capacity }
	capacity_ = max(capacity_, storage.vector_minimum_capacity)
	mut cpu := CpuStorage{
		len: len
		capacity: capacity_
		element_size: element_size
		data: vcalloc(capacity_ * element_size)
	}
	if val != 0 {
		for i in 0 .. cpu.len {
			unsafe { cpu.set_unsafe(i, val) }
		}
	}
	return cpu
}

[unsafe]
fn new_cpu_from_c_array(len int, capacity int, element_size int, c_array voidptr) Storage {
	capacity_ := if capacity < len { len } else { capacity }
	cpu := CpuStorage{
		element_size: element_size
		data: vcalloc(capacity_ * element_size)
		len: len
		capacity: capacity_
	}
	// TODO Write all memory functions (like memcpy) in V
	unsafe { C.memcpy(cpu.data, c_array, len * element_size) }
	return cpu
}

// Private function. Used to implement CpuStorage operator
[unsafe]
fn (cpu CpuStorage) get(i int) voidptr {
	$if !no_bounds_checking ? {
		if i < 0 || i >= cpu.len {
			panic('CpuStorage.get: index out of range (i == $i, cpu.len == $cpu.len)')
		}
	}
	return unsafe { cpu.get_unsafe(i) }
}

// Private function. Used to implement assigment to the CpuStorage element
[unsafe]
fn (mut cpu CpuStorage) set(i int, val voidptr) {
	$if !no_bounds_checking ? {
		if i < 0 || i >= cpu.len {
			panic('CpuStorage.set: index out of range (i == $i, cpu.len == $cpu.len)')
		}
	}
	unsafe { cpu.set_unsafe(i, val) }
}

// fill fills an entire storage with a given value
[unsafe]
fn (mut cpu CpuStorage) fill(val voidptr) {
	for i in 0 .. cpu.len {
		unsafe { cpu.set_unsafe(i, val) }
	}
}

// CpuStorage.clone returns an independent copy of a given CpuStorage
[unsafe]
fn (cpu &CpuStorage) clone() Storage {
	mut size := cpu.capacity * cpu.element_size
	if size == 0 {
		size++
	}
	mut new_cpu := CpuStorage{
		element_size: cpu.element_size
		data: vcalloc(size)
		len: cpu.len
		capacity: cpu.capacity
	}
	// Recursively clone-generated elements if CpuStorage element is CpuStorage type
	size_of_cpu := int(sizeof(CpuStorage))
	if cpu.element_size == size_of_cpu {
		for i in 0 .. cpu.len {
			ar := CpuStorage{}
			unsafe {
				C.memcpy(&ar, cpu.get_unsafe(i), size_of_cpu)
				ar_clone := ar.clone()
				new_cpu.set_unsafe(i, &ar_clone)
			}
		}
	} else {
		unsafe { C.memcpy(byteptr(new_cpu.data), cpu.data, cpu.capacity * cpu.element_size) }
	}
	return new_cpu
}

// CpuStorage.slice returns an CpuStorage using the same buffer as original CpuStorage
// but starting from the `start` element and ending with the element before
// the `end` element of the original CpuStorage with the length and capacity
// set to the number of the elements in the slice.
fn (cpu CpuStorage) slice(start int, _end int) Storage {
	mut end := _end
	$if !no_bounds_checking ? {
		if start > end {
			panic('CpuStorage.slice: invalid slice index ($start > $end)')
		}
		if end > cpu.len {
			panic('CpuStorage.slice: slice bounds out of range ($end >= $cpu.len)')
		}
		if start < 0 {
			panic('CpuStorage.slice: slice bounds out of range ($start < 0)')
		}
	}
	mut data := byteptr(0)
	unsafe {
		data = byteptr(cpu.data) + start * cpu.element_size
	}
	l := end - start
	return CpuStorage{
		element_size: cpu.element_size
		data: data
		len: l
		capacity: l
	}
}

[inline]
fn (cpu CpuStorage) offset(start int) Storage {
	return cpu.slice(start, cpu.len)
}

fn cpu_to_varray<T>(cpu CpuStorage) []T {
	if cpu.element_size == int(sizeof(T)) {
		mut arr := []T{}
		unsafe { arr.push_many(cpu.data, cpu.len) }
		return arr
	}
	panic('CpuStorage.to_varray<T>: incoming type T does not match with the stored data type')
}

// we manually inline this for single operations for performance without -prod
[inline; unsafe]
fn (cpu CpuStorage) get_unsafe(i int) voidptr {
	unsafe {
		return byteptr(cpu.data) + i * cpu.element_size
	}
}

// we manually inline this for single operations for performance without -prod
[inline; unsafe]
fn (mut cpu CpuStorage) set_unsafe(i int, val voidptr) {
	unsafe { C.memcpy(byteptr(cpu.data) + cpu.element_size * i, val, cpu.element_size) }
}

// Apply growth factor if needed
[inline]
fn (mut cpu CpuStorage) ensure_capacity(required int) {
	if required <= cpu.capacity {
		return
	}
	mut capacity := if cpu.capacity < storage.vector_minimum_capacity {
		storage.vector_minimum_capacity
	} else {
		cpu.capacity * storage.vector_growth_factor
	}
	for required > capacity {
		capacity *= storage.vector_growth_factor
	}
	if cpu.capacity == storage.vector_minimum_capacity {
		cpu.data = vcalloc(capacity * cpu.element_size)
	} else {
		unsafe {
			cpu.data = v_realloc(cpu.data, capacity * cpu.element_size)
		}
	}
	cpu.capacity = capacity
}

[inline]
fn max(a int, b int) int {
	return if a > b { a } else { b }
}
