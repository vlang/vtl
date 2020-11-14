module storage

pub const (
        vector_minimum_capacity = 2;
        vector_growth_factor = 2;
        vector_shrink_threshold = 1.0 / 4.0;
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

pub fn new_cpu<T>(capacity int) CpuStorage {
        return CpuStorage{
                len: 0,
                capacity: max(capacity, vector_minimum_capacity),
                data: vcalloc(capacity * int(sizeof(T)))
        }
}

// we manually inline this for single operations for performance without -prod
[inline]
[unsafe]
fn (a CpuStorage) get_unsafe(i int) voidptr {
	unsafe {
		return byteptr(a.data) + i * a.element_size
	}
}

// Private function. Used to implement CpuStorage operator
fn (a CpuStorage) get(i int) voidptr {
	$if !no_bounds_checking ? {
		if i < 0 || i >= a.len {
			panic('CpuStorage.get: index out of range (i == $i, a.len == $a.len)')
		}
	}
	return a.get_unsafe(i)
}

// we manually inline this for single operations for performance without -prod
[inline]
[unsafe]
fn (mut a array) set_unsafe(i int, val voidptr) {
	unsafe {C.memcpy(byteptr(a.data) + a.element_size * i, val, a.element_size)}
}

// Private function. Used to implement assigment to the array element.
fn (mut a array) set(i int, val voidptr) {
	$if !no_bounds_checking ? {
		if i < 0 || i >= a.len {
			panic('CpuStorage.set: index out of range (i == $i, a.len == $a.len)')
		}
	}
	a.set_unsafe(i, val)
}

// Apply growth factor if needed
[inline]
fn (mut a CpuStorage) ensure_capacity(required int) {
	if required <= a.capacity {
		return
	}
	mut capacity := if a.capacity < vector_minimum_capacity {
                vector_minimum_capacity
        } else {
                a.capacity * vector_growth_factor
        }
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
        return if a > b { a } else { b }
}
