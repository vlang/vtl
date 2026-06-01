module autograd_cuda

import vsl.cuda
import vsl.cuda.compute
import vtl
import vtl.la

// linear_backward_f64 runs Linear gradients; uses cuBLAS GEMM when cuda_backward_enabled.
pub fn linear_backward_f64(grad &vtl.Tensor[f64], input &vtl.Tensor[f64], weight &vtl.Tensor[f64],
	bias_needs_grad bool, mut session DeviceSession) ![]&vtl.Tensor[f64] {
	if !cuda_backward_enabled() || !session.enabled {
		return linear_backward_f64_cpu(grad, input, weight, bias_needs_grad)
	}
	dev := cuda.get_default_device()!
	g := grad.to_array()
	in_a := input.to_array()
	w_a := weight.to_array()
	m := grad.shape[0]
	n := grad.shape[1]
	k := input.shape[1]

	d_in := compute.gemm_cuda(dev, g, w_a, m, k, n)!
	grad_t := transpose_row(g, m, n)
	d_w := compute.gemm_cuda(dev, grad_t, in_a, n, k, m)!
	d_in_t := vtl.from_array(d_in, [m, k])!
	d_w_t := vtl.from_array(d_w, [n, k])!
	mut result := [d_in_t, d_w_t, grad]
	if bias_needs_grad {
		ones := vtl.ones[f64]([1, m])
		d_b := compute.gemm_cuda(dev, ones.to_array(), g, 1, n, m)!
		result[2] = vtl.from_array(d_b, [1, n])!
	}
	return result
}

fn transpose_row(a []f64, rows int, cols int) []f64 {
	mut t := []f64{len: rows * cols}
	for r in 0 .. rows {
		for c in 0 .. cols {
			t[c * rows + r] = a[r * cols + c]
		}
	}
	return t
}
