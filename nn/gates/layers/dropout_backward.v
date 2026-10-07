module layers

import vtl

// dropout_gate_backward_f64_cpu applies the saved dropout mask and inverted
// keep probability to an upstream gradient.
fn dropout_gate_backward_f64_cpu(gradient &vtl.Tensor[f64], mask &vtl.Tensor[f64], keep_prob f64) !&vtl.Tensor[f64] {
	if gradient.shape != mask.shape {
		return error('dropout backward: gradient and mask shapes must match')
	}
	if keep_prob <= 0 || keep_prob > 1 {
		return error('dropout backward: keep probability must be in (0, 1]')
	}
	return gradient.nmap([mask], fn [keep_prob] (xs []f64, _ []int) f64 {
		return xs[0] * xs[1] / keep_prob
	})
}
