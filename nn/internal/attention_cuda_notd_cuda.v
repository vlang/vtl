module internal

import vtl

// attention_cuda_eligible is always false without `-d cuda`, so the CPU path runs.
pub fn attention_cuda_eligible() bool {
	return false
}

// attention_backward_cuda_f64 exists only so the layer dispatch compiles without `-d cuda`.
pub fn attention_backward_cuda_f64(grad &vtl.Tensor[f64], input &vtl.Tensor[f64], w_q &vtl.Tensor[f64], w_k &vtl.Tensor[f64], w_v &vtl.Tensor[f64], w_o &vtl.Tensor[f64], q &vtl.Tensor[f64], k &vtl.Tensor[f64], v &vtl.Tensor[f64], attn_weights &vtl.Tensor[f64], merged &vtl.Tensor[f64], num_heads int, head_dim int) ![]&vtl.Tensor[f64] {
	return error('attention cuBLAS backward: build with -d cuda')
}
