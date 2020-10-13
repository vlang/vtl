module num

import vsl.math

//
// Buffer is a generic interface to represent data storage
//
// for an ndarray.
// interface Buffer {
// ptr          ()&f64
// stride_offset(shape []int, strides []int)&f64
// offset       (start int)Buffer
// set          (index []int, strides []int, value f64)
// get          (index []int, strides []int)f64
// }
// Storage for computations run on a CPU
pub struct CpuStorage {
	buffer voidptr
}

pub fn (c CpuStorage) str() string {
	return '<CPU Storage>'
}

// ptr returns the raw pointer for a buffer's data
pub fn (c CpuStorage) ptr() &f64 {
	return &f64(c.buffer)
}

// stride offset returns a raw pointer representing the starting offset
// for arrays that may be negatively strided
pub fn (c CpuStorage) stride_offset(shape, strides []int) &f64 {
	mut ptr := c.ptr()
	for i := 0; i < shape.len; i++ {
		if strides[i] < 0 {
			unsafe {
				ptr += (shape[i] - 1) * int(math.abs(strides[i]))
			}
		}
	}
	return ptr
}

pub fn cpu_from_pointer(ptr &f64) CpuStorage {
	return CpuStorage{
		buffer: ptr
	}
}

// offset returns a new buffer offset by a particular amount.
// This makes slicing operations trivial when returning a new
// buffer for the child array
pub fn (c CpuStorage) offset(start int) CpuStorage {
	unsafe {
		buffer := c.ptr() + start
		return CpuStorage{
			buffer: buffer
		}
	}
}

// set sets a value given a provided index and strides.
pub fn (c CpuStorage) set(index, shape, strides []int, value f64) {
	mut raw := c.stride_offset(shape, strides)
	for i := 0; i < shape.len; i++ {
		mut idxer := index[i]
		if idxer < 0 {
			idxer += shape[i]
		}
		unsafe {
			raw += idxer * strides[i]
		}
	}
	unsafe {
		*raw = value
	}
}

// get returns a value given a provided index and strides
pub fn (c CpuStorage) get(index, shape, strides []int) f64 {
	mut raw := c.stride_offset(shape, strides)
	for i := 0; i < shape.len; i++ {
		mut idxer := index[i]
		if idxer < 0 {
			idxer += shape[i]
		}
		unsafe {
			raw += idxer * strides[i]
		}
	}
	return *raw
}

// allocates storage for use with computations on a GPU
pub fn cpu(size int) CpuStorage {
	unsafe {
		buffer := []f64{len: size}
		return CpuStorage{
			buffer: &f64(buffer.data)
		}
	}
}

pub fn cpu_from_array(arr []f64) CpuStorage {
	buffer := arr.data
	return CpuStorage{
		buffer: buffer
	}
}
