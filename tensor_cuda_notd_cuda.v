module vtl

import vtl.storage as _

@[params]
pub struct CudaParams {}

// cuda returns an error when CUDA is not enabled at compile time
pub fn (t &Tensor[T]) cuda(params CudaParams) !&Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d cuda" to use the CUDA Backend')
}
