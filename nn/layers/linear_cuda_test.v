module layers

import vtl
import vtl.autograd
import vtl.autograd_cuda
import math

fn test_linear_forward_cpu_without_cuda_env() ! {
	// Default: no VTL_USE_CUDA — must match CPU matmul path
	mut c := autograd.ctx[f64]()
	autograd_cuda.attach_context_session(mut c)
	layer := linear_layer[f64](c, 4, 2)
	input := vtl.from_array([1.0, 0.0, -1.0, 0.5], [1, 4])!
	x := c.variable(input)
	out := layer.forward(x)!
	assert out.value.shape == [1, 2]
}

fn test_linear_cuda_matches_cpu_reference() ! {
	if !cuda_tests_enabled() {
		return
	}
	mut c := autograd.ctx[f64]()
	autograd_cuda.attach_context_session(mut c)
	layer_iface := linear_layer[f64](c, 3, 2)
	input := vtl.from_array([1.0, 2.0, 3.0], [1, 3])!
	vars := layer_iface.variables()
	w := vars[0].value
	bias_t := vars[1].value

	cpu := linear_forward_f64(input, w, bias_t, unsafe { nil }, unsafe { nil })!
	gpu := linear_forward_cuda_f64(input, w, bias_t)!

	assert cpu.shape == gpu.shape
	for i in 0 .. cpu.size {
		diff := math.abs(f64(cpu.get_nth(i)) - f64(gpu.get_nth(i)))
		assert diff < 1e-9, 'CPU vs CUDA linear mismatch at ${i}: ${diff}'
	}
}

fn test_linear_layer_forward_with_cuda_env() ! {
	if !cuda_tests_enabled() {
		return
	}
	mut c := autograd.ctx[f64]()
	autograd_cuda.attach_context_session(mut c)
	layer := linear_layer[f64](c, 4, 3)
	input := vtl.from_array([1.0, 0.0, -1.0, 0.5], [1, 4])!
	x := c.variable(input)
	out := layer.forward(x)!
	assert out.value.shape == [1, 3]
}
