module layers

import vtl

// dropout_gate_backward_f64 keeps the CPU implementation available without CUDA.
fn dropout_gate_backward_f64(gradient &vtl.Tensor[f64], mask &vtl.Tensor[f64], keep_prob f64) !&vtl.Tensor[f64] {
	return dropout_gate_backward_f64_cpu(gradient, mask, keep_prob)
}
