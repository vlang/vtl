module storage

@[params]
pub struct CudaParams {
	device_id int
	stream    voidptr = unsafe { nil }
}

// cuda returns an error when CUDA is not enabled at compile time
pub fn (cpu &CpuStorage[T]) cuda(params CudaParams) !&CpuStorage[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d cuda" to use the CUDA Backend')
}
