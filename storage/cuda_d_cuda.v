module storage

import vsl.cuda

@[params]
pub struct CudaParams {
	device_id int = 0
}

// CudaStorage holds tensor data on GPU memory
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
		C.cudaMemcpy(ptr, arr.data, sz, C.cuda_memcpy_host_to_device)
	}

	return &CudaStorage[T]{
		device: device
		ptr:    ptr
		size:   sz
		count:  count
	}
}

// from_cuda returns the same CudaStorage (identity function for chaining)
@[inline]
pub fn (storage &CudaStorage[T]) cuda(params CudaParams) !&CudaStorage[T] {
	return storage
}

// cpu transfers data from GPU to CPU memory
pub fn (storage &CudaStorage[T]) cpu() !&CpuStorage[T] {
	if isnil(storage.ptr) {
		return error('CudaStorage.cpu: null pointer')
	}
	mut arr := []T{len: storage.count}
	sz := int(sizeof(T)) * storage.count
	status := C.cudaMemcpy(arr.data, storage.ptr, sz, C.cuda_memcpy_device_to_host)
	if status != 0 {
		return error('CudaStorage.cpu: cudaMemcpy failed with status ${status}')
	}
	return &CpuStorage[T]{
		data: arr
	}
}

// to_array transfers data from GPU to a V array
pub fn (storage &CudaStorage[T]) to_array() ![]T {
	if isnil(storage.ptr) {
		return error('CudaStorage.to_array: null pointer')
	}
	mut arr := []T{len: storage.count}
	sz := int(sizeof(T)) * storage.count
	status := C.cudaMemcpy(arr.data, storage.ptr, sz, C.cuda_memcpy_device_to_host)
	if status != 0 {
		return error('CudaStorage.to_array: cudaMemcpy failed with status ${status}')
	}
	return arr
}

// release releases the GPU memory
pub fn (storage &CudaStorage[T]) release() {
	if !isnil(storage.ptr) {
		C.cudaFree(storage.ptr)
	}
}

// device returns the CUDA device associated with this storage
pub fn (storage &CudaStorage[T]) device() &cuda.CudaDevice {
	return storage.device
}
