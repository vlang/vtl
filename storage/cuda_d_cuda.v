module storage

import vsl.cuda

// CudaParams defines a public data structure for this module.

// CudaParams defines a public data structure for this module.
@[params]
pub struct CudaParams {
	device_id int = 0
}

// CudaStorage holds tensor data on GPU memory

// CudaStorage defines a public data structure for this module.

// CudaStorage defines a public data structure for this module.
@[heap]
pub struct CudaStorage[T] {
pub mut:
	// device is the CUDA device this storage is allocated on
	device &cuda.CudaDevice = unsafe { nil }
	// ptr is the raw GPU memory pointer
	ptr voidptr
	// size is the total byte size
	size int
	// count is the number of elements of type T
	count int
}

// from_cpu creates a CudaStorage from CPU memory with GPU allocation and copy
pub fn (cpu &CpuStorage[T]) cuda(params CudaParams) !&CudaStorage[T] {
	device := cuda.get_device(params.device_id)!

	arr := cpu.data
	count := arr.len
	mut ptr := unsafe { nil }
	sz := int(sizeof(T)) * count
	status := C.cudaMalloc(&ptr, sz)
	if status != 0 {
		return error('CudaStorage.cuda: cudaMalloc failed with status ${status}')
	}

	// Copy data from CPU to GPU
	unsafe {
		C.cudaMemcpy(ptr, arr.data, sz, cuda.cuda_memcpy_host_to_device)
	}

	return &CudaStorage[T]{
		device: device
		ptr:    ptr
		size:   sz
		count:  count
	}
}

// from_cuda returns the same CudaStorage (identity function for chaining)

// cuda exposes this operation as part of the public API.

// cuda exposes this operation as part of the public API.
@[inline]
pub fn (cstorage &CudaStorage[T]) cuda(params CudaParams) !&CudaStorage[T] {
	return cstorage
}

// cpu transfers data from GPU to CPU memory
pub fn (cstorage &CudaStorage[T]) cpu() !&CpuStorage[T] {
	if isnil(cstorage.ptr) {
		return error('CudaStorage.cpu: null pointer')
	}
	mut arr := []T{len: cstorage.count}
	sz := int(sizeof(T)) * cstorage.count
	status := C.cudaMemcpy(arr.data, cstorage.ptr, sz, cuda.cuda_memcpy_device_to_host)
	if status != 0 {
		return error('CudaStorage.cpu: cudaMemcpy failed with status ${status}')
	}
	return &CpuStorage[T]{
		data: arr
	}
}

// to_array transfers data from GPU to a V array
pub fn (cstorage &CudaStorage[T]) to_array() ![]T {
	if isnil(cstorage.ptr) {
		return error('CudaStorage.to_array: null pointer')
	}
	mut arr := []T{len: cstorage.count}
	sz := int(sizeof(T)) * cstorage.count
	status := C.cudaMemcpy(arr.data, cstorage.ptr, sz, cuda.cuda_memcpy_device_to_host)
	if status != 0 {
		return error('CudaStorage.to_array: cudaMemcpy failed with status ${status}')
	}
	return arr
}

// release releases the GPU memory
pub fn (cstorage &CudaStorage[T]) release() {
	if !isnil(cstorage.ptr) {
		C.cudaFree(cstorage.ptr)
	}
}

// device returns the CUDA device associated with this storage
pub fn (cstorage &CudaStorage[T]) device() &cuda.CudaDevice {
	return cstorage.device
}
