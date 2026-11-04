module internal

import math
import os
import vtl

// Gradient check: cuDNN BatchNorm2D backward against the CPU reference.
// Runs only with `-d cuda` and VTL_TEST_CUDA=1 (same skip pattern as conv2d_cuda_test.v).
fn batchnorm2d_cuda_test_fixture() !(&vtl.Tensor[f64], &vtl.Tensor[f64], &vtl.Tensor[f64], &vtl.Tensor[f64]) {
	mut input_values := []f64{len: 4 * 3 * 5 * 5}
	for i in 0 .. input_values.len {
		input_values[i] = f64(i % 13 - 6) * 0.37 + f64(i) * 0.011
	}
	mut grad_values := []f64{len: input_values.len}
	for i in 0 .. grad_values.len {
		grad_values[i] = f64(i % 7 - 3) * 0.5 + 0.1
	}
	input := vtl.from_array(input_values, [4, 3, 5, 5])!
	grad := vtl.from_array(grad_values, [4, 3, 5, 5])!
	gamma := vtl.from_array([f64(1.2), -0.6, 0.9], [3])!
	beta := vtl.from_array([f64(0.1), -0.3, 0.25], [3])!
	return input, grad, gamma, beta
}

fn assert_slices_close(got []f64, want []f64, tol f64, label string) {
	assert got.len == want.len, '${label}: length ${got.len} != ${want.len}'
	for i in 0 .. got.len {
		assert math.abs(got[i] - want[i]) < tol, '${label}[${i}]: got ${got[i]}, want ${want[i]}'
	}
}

fn test_batchnorm2d_backward_cuda_matches_cpu_reference() ! {
	$if !cuda {
		return
	}
	if os.getenv('VTL_TEST_CUDA') != '1' {
		return
	}
	eps := 1e-5
	input, grad, gamma, beta := batchnorm2d_cuda_test_fixture()!
	_, mean, var_ := batchnorm2d_training[f64](input, gamma, beta, eps)!

	cpu := batchnorm2d_backward_cpu[f64](grad, input, gamma, beta, mean, var_, eps, true)!
	gpu := batchnorm2d_backward_cuda_f64(grad, input, gamma, mean, var_, eps)!

	assert cpu.len == 3
	assert gpu.len == 3
	assert gpu[0].shape == input.shape
	assert gpu[1].shape == [3]
	assert gpu[2].shape == [3]
	names := ['d_input', 'd_gamma', 'd_beta']
	for k in 0 .. 3 {
		assert_slices_close(gpu[k].to_array(), cpu[k].to_array(), 1e-9, names[k])
	}
}

fn test_batchnorm2d_backward_dispatch_selects_cuda_when_enabled() ! {
	$if !cuda {
		return
	}
	if os.getenv('VTL_TEST_CUDA') != '1' {
		return
	}
	eps := 1e-5
	input, grad, gamma, beta := batchnorm2d_cuda_test_fixture()!
	_, mean, var_ := batchnorm2d_training[f64](input, gamma, beta, eps)!

	os.setenv('VTL_USE_CUDA', '1', true)
	os.setenv('VTL_CUDA_BACKWARD', '1', true)
	defer {
		os.unsetenv('VTL_USE_CUDA')
		os.unsetenv('VTL_CUDA_BACKWARD')
	}
	$if cuda ? {
		assert batchnorm_cudnn_eligible(eps, true)
		assert !batchnorm_cudnn_eligible(eps, false)
		assert !batchnorm_cudnn_eligible(1e-6, true)
	}

	dispatched := batchnorm2d_backward_mode[f64](grad, input, gamma, beta, mean, var_, eps, true)!
	cpu := batchnorm2d_backward_cpu[f64](grad, input, gamma, beta, mean, var_, eps, true)!
	names := ['d_input', 'd_gamma', 'd_beta']
	for k in 0 .. 3 {
		assert_slices_close(dispatched[k].to_array(), cpu[k].to_array(), 1e-9, names[k])
	}
}
