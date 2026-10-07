module layers

import math
import os
import vtl
import vtl.autograd

fn test_dropout_gate_f64_backward_applies_mask_and_keep_probability() ! {
	mask := vtl.from_array([1.0, 0.0, 1.0, 0.0], [2, 2])!
	gradient := vtl.from_array([2.0, 4.0, -3.0, 8.0], [2, 2])!
	ctx := autograd.ctx[f64]()
	mut output := ctx.variable(vtl.zeros[f64]([2, 2]))
	output.grad = gradient
	gate := dropout_gate[f64](mask, 0.5)
	result := gate.backward(autograd.payload(output))!
	assert result[0].to_array() == [4.0, 0.0, -6.0, 0.0]
}

fn test_dropout_gate_f32_backward_remains_cpu() ! {
	mask := vtl.from_array([f32(1), 0, 1, 0], [2, 2])!
	gradient := vtl.from_array([f32(2), 4, -3, 8], [2, 2])!
	ctx := autograd.ctx[f32]()
	mut output := ctx.variable(vtl.zeros[f32]([2, 2]))
	output.grad = gradient
	assert output.grad.to_array() == [f32(2), 4, -3, 8]
	gate := dropout_gate[f32](mask, 0.5)
	result := gate.backward(autograd.payload(output))!
	assert result[0].to_array() == [f32(4), 0, -6, 0]
}

fn test_dropout_backward_f64_rejects_invalid_inputs() ! {
	gradient := vtl.from_array([1.0, 2.0], [1, 2])!
	wrong_shape := vtl.from_array([1.0, 0.0], [2, 1])!
	if _ := dropout_gate_backward_f64_cpu(gradient, wrong_shape, 0.5) {
		assert false, 'expected a shape mismatch error'
	}
	mask := vtl.from_array([1.0, 0.0], [1, 2])!
	if _ := dropout_gate_backward_f64_cpu(gradient, mask, 0.0) {
		assert false, 'expected an invalid keep probability error'
	}
}

fn test_dropout_backward_f64_cuda_matches_cpu_when_enabled() ! {
	if os.getenv('VTL_TEST_CUDA') != '1' || os.getenv('VTL_CUDA_BACKWARD') != '1' {
		return
	}
	gradient := vtl.from_array([2.0, 4.0, -3.0, 8.0], [2, 2])!
	mask := vtl.from_array([1.0, 0.0, 1.0, 0.0], [2, 2])!
	cpu := dropout_gate_backward_f64_cpu(gradient, mask, 0.5)!
	gpu := dropout_gate_backward_f64(gradient, mask, 0.5)!
	for i in 0 .. cpu.size {
		assert math.abs(cpu.get_nth(i) - gpu.get_nth(i)) < 1e-12
	}
}
