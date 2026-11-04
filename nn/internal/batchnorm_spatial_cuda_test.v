module internal

import math
import os
import vtl

// Gradient checks: cuDNN BatchNorm1D and BatchNorm3D backward against the CPU reference.
// Runs only with `-d cuda` and VTL_TEST_CUDA=1, the same skip pattern as
// batchnorm2d_backward_cuda_test.v. The CPU reference is the default dispatch
// without VTL_USE_CUDA, so it is computed before the environment is set.

fn assert_spatial_slices_close(got []f64, want []f64, tol f64, label string) {
	assert got.len == want.len, '${label}: length ${got.len} != ${want.len}'
	for i in 0 .. got.len {
		assert math.abs(got[i] - want[i]) < tol, '${label}[${i}]: got ${got[i]}, want ${want[i]}'
	}
}

// disable_cuda_env keeps the reference computation on the CPU path.
fn disable_cuda_env() {
	os.unsetenv('VTL_USE_CUDA')
	os.unsetenv('VTL_CUDA_BACKWARD')
}

fn batchnorm_spatial_cuda_test_skip() bool {
	$if !cuda {
		return true
	}
	return os.getenv('VTL_TEST_CUDA') != '1'
}

fn batchnorm1d_cuda_test_fixture() !(&vtl.Tensor[f64], &vtl.Tensor[f64], &vtl.Tensor[f64], &vtl.Tensor[f64]) {
	mut input_values := []f64{len: 8 * 6}
	for i in 0 .. input_values.len {
		input_values[i] = f64(i % 11 - 5) * 0.41 + f64(i) * 0.013
	}
	mut grad_values := []f64{len: input_values.len}
	for i in 0 .. grad_values.len {
		grad_values[i] = f64(i % 5 - 2) * 0.6 + 0.05
	}
	input := vtl.from_array(input_values, [8, 6])!
	grad := vtl.from_array(grad_values, [8, 6])!
	gamma := vtl.from_array([f64(1.1), -0.5, 0.8, 1.4, -1.0, 0.3], [1, 6])!
	beta := vtl.from_array([f64(0.2), -0.1, 0.05, 0.4, -0.3, 0.0], [1, 6])!
	return input, grad, gamma, beta
}

fn batchnorm3d_cuda_test_fixture() !(&vtl.Tensor[f64], &vtl.Tensor[f64], &vtl.Tensor[f64], &vtl.Tensor[f64]) {
	mut input_values := []f64{len: 3 * 4 * 2 * 3 * 4}
	for i in 0 .. input_values.len {
		input_values[i] = f64(i % 17 - 8) * 0.29 + f64(i) * 0.007
	}
	mut grad_values := []f64{len: input_values.len}
	for i in 0 .. grad_values.len {
		grad_values[i] = f64(i % 9 - 4) * 0.35 + 0.08
	}
	input := vtl.from_array(input_values, [3, 4, 2, 3, 4])!
	grad := vtl.from_array(grad_values, [3, 4, 2, 3, 4])!
	gamma := vtl.from_array([f64(0.9), -0.7, 1.3, 0.5], [4])!
	beta := vtl.from_array([f64(-0.2), 0.15, 0.0, 0.35], [4])!
	return input, grad, gamma, beta
}

fn test_batchnorm1d_backward_cuda_matches_cpu_reference() ! {
	if batchnorm_spatial_cuda_test_skip() {
		return
	}
	eps := 1e-5
	input, grad, gamma, beta := batchnorm1d_cuda_test_fixture()!
	_, mean, var_ := batchnorm1d_training[f64](input, gamma, beta, eps)!

	disable_cuda_env()
	cpu := batchnorm1d_backward_mode[f64](grad, input, gamma, beta, mean, var_, eps, true)!
	gpu := batchnorm1d_backward_cuda_f64(grad, input, gamma, mean, var_, eps)!

	assert gpu.len == 3
	assert gpu[0].shape == input.shape
	assert gpu[1].shape == [1, 6]
	assert gpu[2].shape == [1, 6]
	names := ['d_input', 'd_gamma', 'd_beta']
	for k in 0 .. 3 {
		assert_spatial_slices_close(gpu[k].to_array(), cpu[k].to_array(), 1e-9, names[k])
	}
}

fn test_batchnorm3d_backward_cuda_matches_cpu_reference() ! {
	if batchnorm_spatial_cuda_test_skip() {
		return
	}
	eps := 1e-5
	input, grad, gamma, beta := batchnorm3d_cuda_test_fixture()!
	_, mean, var_ := batchnorm3d_training[f64](input, gamma, beta, eps)!

	disable_cuda_env()
	cpu := batchnorm3d_backward_mode[f64](grad, input, gamma, beta, mean, var_, eps, true)!
	gpu := batchnorm3d_backward_cuda_f64(grad, input, gamma, mean, var_, eps)!

	assert gpu.len == 3
	assert gpu[0].shape == input.shape
	assert gpu[1].shape == [4]
	assert gpu[2].shape == [4]
	names := ['d_input', 'd_gamma', 'd_beta']
	for k in 0 .. 3 {
		assert_spatial_slices_close(gpu[k].to_array(), cpu[k].to_array(), 1e-9, names[k])
	}
}

fn test_batchnorm1d_and_3d_dispatch_select_cuda_when_enabled() ! {
	if batchnorm_spatial_cuda_test_skip() {
		return
	}
	eps := 1e-5
	input1, grad1, gamma1, beta1 := batchnorm1d_cuda_test_fixture()!
	_, mean1, var1 := batchnorm1d_training[f64](input1, gamma1, beta1, eps)!
	input3, grad3, gamma3, beta3 := batchnorm3d_cuda_test_fixture()!
	_, mean3, var3 := batchnorm3d_training[f64](input3, gamma3, beta3, eps)!

	disable_cuda_env()
	cpu1 := batchnorm1d_backward_mode[f64](grad1, input1, gamma1, beta1, mean1, var1, eps, true)!
	cpu3 := batchnorm3d_backward_mode[f64](grad3, input3, gamma3, beta3, mean3, var3, eps, true)!

	os.setenv('VTL_USE_CUDA', '1', true)
	os.setenv('VTL_CUDA_BACKWARD', '1', true)
	defer {
		os.unsetenv('VTL_USE_CUDA')
		os.unsetenv('VTL_CUDA_BACKWARD')
	}
	assert batchnorm_cudnn_eligible(eps, true)
	assert !batchnorm_cudnn_eligible(eps, false)

	dispatched1 := batchnorm1d_backward_mode[f64](grad1, input1, gamma1, beta1, mean1, var1, eps, true)!
	dispatched3 := batchnorm3d_backward_mode[f64](grad3, input3, gamma3, beta3, mean3, var3, eps, true)!
	names := ['d_input', 'd_gamma', 'd_beta']
	for k in 0 .. 3 {
		assert_spatial_slices_close(dispatched1[k].to_array(), cpu1[k].to_array(), 1e-9, names[k])
		assert_spatial_slices_close(dispatched3[k].to_array(), cpu3[k].to_array(), 1e-9, names[k])
	}
}
