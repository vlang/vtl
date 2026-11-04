module internal

import vtl

// embedding_cuda_eligible is always false without `-d cuda`, so the CPU path runs.
fn embedding_cuda_eligible() bool {
	return false
}

// embedding_backward_cuda_f64 exists only so the generic dispatch compiles without `-d cuda`.
fn embedding_backward_cuda_f64(grad_out &vtl.Tensor[f64], input &vtl.Tensor[f64], weight &vtl.Tensor[f64]) ![]&vtl.Tensor[f64] {
	return error('embedding cuBLAS backward: build with -d cuda')
}
