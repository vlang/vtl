module internal

import math
import os
import vtl

// Gradient checks: cuBLAS Embedding backward against the CPU reference.
// Runs only with `-d cuda` and VTL_TEST_CUDA=1, the same skip pattern as the other CUDA
// tests. The CPU reference is computed with the CUDA environment cleared.

fn embedding_cuda_test_skip() bool {
	$if !cuda {
		return true
	}
	return os.getenv('VTL_TEST_CUDA') != '1'
}

fn embedding_cuda_disable_env() {
	os.unsetenv('VTL_USE_CUDA')
	os.unsetenv('VTL_CUDA_BACKWARD')
}

fn assert_embedding_slices_close(got []f64, want []f64, tol f64, label string) {
	assert got.len == want.len, '${label}: length ${got.len} != ${want.len}'
	for i in 0 .. got.len {
		assert math.abs(got[i] - want[i]) < tol, '${label}[${i}]: got ${got[i]}, want ${want[i]}'
	}
}

// embedding_cuda_fixture uses repeated indices and one out-of-range index (-1), which
// both paths must ignore.
fn embedding_cuda_fixture() !(&vtl.Tensor[f64], &vtl.Tensor[f64], &vtl.Tensor[f64]) {
	input := vtl.from_array([f64(0), 2, 2, 5, -1, 0], [2, 3])!
	mut grad_values := []f64{len: 2 * 3 * 4}
	for i in 0 .. grad_values.len {
		grad_values[i] = f64(i % 9 - 4) * 0.3 + 0.05
	}
	grad := vtl.from_array(grad_values, [2, 3, 4])!
	mut weight_values := []f64{len: 6 * 4}
	for i in 0 .. weight_values.len {
		weight_values[i] = f64(i) * 0.01
	}
	weight := vtl.from_array(weight_values, [6, 4])!
	return grad, input, weight
}

fn test_embedding_backward_cuda_matches_cpu_reference() ! {
	if embedding_cuda_test_skip() {
		return
	}
	grad, input, weight := embedding_cuda_fixture()!
	embedding_cuda_disable_env()
	cpu := embedding_backward[f64](grad, input, weight)!
	gpu := embedding_backward_cuda_f64(grad, input, weight)!

	assert gpu.len == 1
	assert gpu[0].shape == weight.shape
	assert_embedding_slices_close(gpu[0].to_array(), cpu[0].to_array(), 1e-9, 'd_weight')
}

fn test_embedding_backward_dispatch_selects_cuda_when_enabled() ! {
	if embedding_cuda_test_skip() {
		return
	}
	grad, input, weight := embedding_cuda_fixture()!
	embedding_cuda_disable_env()
	cpu := embedding_backward[f64](grad, input, weight)!

	os.setenv('VTL_USE_CUDA', '1', true)
	os.setenv('VTL_CUDA_BACKWARD', '1', true)
	defer {
		embedding_cuda_disable_env()
	}
	assert embedding_cuda_eligible()
	dispatched := embedding_backward[f64](grad, input, weight)!
	assert_embedding_slices_close(dispatched[0].to_array(), cpu[0].to_array(), 1e-9, 'd_weight')
}
