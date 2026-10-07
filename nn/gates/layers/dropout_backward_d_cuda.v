module layers

import vsl.cuda
import vsl.cuda.compute
import vtl

// dropout_gate_backward_f64 uses cuBLAS for the masked gradient when enabled.
fn dropout_gate_backward_f64(gradient &vtl.Tensor[f64], mask &vtl.Tensor[f64], keep_prob f64) !&vtl.Tensor[f64] {
	if !linear_gate_use_cuda_backward() {
		return dropout_gate_backward_f64_cpu(gradient, mask, keep_prob)
	}
	if gradient.shape != mask.shape {
		return error('dropout backward: gradient and mask shapes must match')
	}
	if keep_prob <= 0 || keep_prob > 1 {
		return error('dropout backward: keep probability must be in (0, 1]')
	}
	dev := cuda.get_default_device()!
	mut values := compute.mul_vec_cuda(dev, gradient.to_array(), mask.to_array())!
	for i in 0 .. values.len {
		values[i] /= keep_prob
	}
	return vtl.from_array(values, gradient.shape)
}
