module layers

import os
import vtl
import vtl.autograd
import vtl.autograd_cuda

$if cuda ? {
	// Chained Linear forwards reuse device activations when VTL_GPU_ACTIVATIONS=1 (`-d cuda` only).
	fn test_linear_gpu_activation_chain_two_layers() ! {
		if !cuda_tests_enabled() || os.getenv('VTL_GPU_ACTIVATIONS') != '1' {
			return
		}
		mut c := autograd.ctx[f64]()
		autograd_cuda.attach_context_session(mut c)
		l1 := linear_layer[f64](c, 3, 2)
		l2 := linear_layer[f64](c, 2, 4)
		inp := vtl.from_array([1.0, 2.0, 3.0], [1, 3])!
		x := c.variable(inp)
		h := l1.forward(x)!
		assert h.has_gpu_activation(), 'first Linear should bind GPU activation'
		out := l2.forward(h)!
		assert out.value.shape == [1, 4]
		assert !h.has_gpu_activation(), 'second Linear should take GPU activation from input'
	}
}
