module internal

import vtl

// lstm_cuda_eligible is always false without `-d cuda`, so the CPU path runs.
pub fn lstm_cuda_eligible() bool {
	return false
}

// lstm_backward_cuda_f64 exists only so the generic dispatch compiles without `-d cuda`.
fn lstm_backward_cuda_f64(input &vtl.Tensor[f64], hidden0 &vtl.Tensor[f64], cell0 &vtl.Tensor[f64], w_ih &vtl.Tensor[f64], w_hh &vtl.Tensor[f64], b_ih &vtl.Tensor[f64], b_hh &vtl.Tensor[f64], grad_output &vtl.Tensor[f64], grad_final_hidden &vtl.Tensor[f64], grad_final_cell &vtl.Tensor[f64]) ![]&vtl.Tensor[f64] {
	return error('lstm cuBLAS backward: build with -d cuda')
}
