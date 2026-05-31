module layers

import vtl

// linear_forward_cuda_f64 stub when not built with `-d cuda`.
pub fn linear_forward_cuda_f64(x &vtl.Tensor[f64], weights &vtl.Tensor[f64], bias &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	return error(@FN + ': compile with -d cuda to use CUDA linear forward')
}

// relu_forward_cuda returns error if CUDA is not enabled
pub fn relu_forward_cuda(x &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	return error(@FN + ': compile with -d cuda to use the CUDA backend')
}

// sigmoid_forward_cuda returns error if CUDA is not enabled
pub fn sigmoid_forward_cuda(x &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	return error(@FN + ': compile with -d cuda to use the CUDA backend')
}
