module layers

@[params]
pub struct CudaParams {}

// linear_forward_cuda returns error if CUDA is not enabled
pub fn linear_forward_cuda[T](x &Tensor[T], weights &Tensor[T], bias &Tensor[T]) !&Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d cuda" to use the CUDA Backend')
}

// relu_forward_cuda returns error if CUDA is not enabled
pub fn relu_forward_cuda[T](x &Tensor[T]) !&Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d cuda" to use the CUDA Backend')
}

// sigmoid_forward_cuda returns error if CUDA is not enabled
pub fn sigmoid_forward_cuda[T](x &Tensor[T]) !&Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d cuda" to use the CUDA Backend')
}
