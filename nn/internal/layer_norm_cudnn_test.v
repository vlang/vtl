module internal

import math
import os
import vtl

// Gradient checks: cuDNN LayerNorm backward (graph API, FP32) against the CPU reference.
// Tolerance is 1e-4: FP32 storage on the device, while the CPU path accumulates in FP64.
// Runs only with `-d cuda` and VTL_TEST_CUDA=1, the same skip pattern as the
// BatchNorm CUDA tests. The CPU reference is computed with the CUDA environment
// cleared, so it cannot be silently dispatched to the GPU.

fn ln_cudnn_test_skip() bool {
	$if !cuda {
		return true
	}
	return os.getenv('VTL_TEST_CUDA') != '1'
}

fn ln_cudnn_disable_env() {
	os.unsetenv('VTL_USE_CUDA')
	os.unsetenv('VTL_CUDA_BACKWARD')
}

fn assert_ln_slices_close(got []f32, want []f32, tol f64, label string) {
	assert got.len == want.len, '${label}: length ${got.len} != ${want.len}'
	for i in 0 .. got.len {
		assert math.abs(f64(got[i]) - f64(want[i])) < tol, '${label}[${i}]: got ${got[i]}, want ${want[i]}'
	}
}

// ln_cudnn_fixture builds input, gradient, gamma and beta for a LayerNorm over the
// trailing normalized_shape block of input_shape.
fn ln_cudnn_fixture(input_shape []int, normalized_shape []int) !(&vtl.Tensor[f32], &vtl.Tensor[f32], &vtl.Tensor[f32], &vtl.Tensor[f32]) {
	mut size := 1
	for dim in input_shape {
		size *= dim
	}
	mut input_values := []f32{len: size}
	mut grad_values := []f32{len: size}
	for i in 0 .. size {
		input_values[i] = f32(f64(i % 13 - 6) * 0.37 + f64(i) * 0.011)
		grad_values[i] = f32(f64(i % 7 - 3) * 0.5 + 0.1)
	}
	mut affine_size := 1
	for dim in normalized_shape {
		affine_size *= dim
	}
	mut gamma_values := []f32{len: affine_size}
	mut beta_values := []f32{len: affine_size}
	for i in 0 .. affine_size {
		gamma_values[i] = f32(0.8 + f64(i) * 0.15)
		beta_values[i] = f32(-0.2 + f64(i) * 0.05)
	}
	input := vtl.from_array(input_values, input_shape)!
	grad := vtl.from_array(grad_values, input_shape)!
	gamma := vtl.from_array(gamma_values, normalized_shape)!
	beta := vtl.from_array(beta_values, normalized_shape)!
	return input, grad, gamma, beta
}

fn test_layer_norm_cudnn_backward_matches_cpu_reference() ! {
	if ln_cudnn_test_skip() {
		return
	}
	eps := 1e-5
	cases := [
		[[5, 4], [4]],
		[[4, 2, 3], [2, 3]],
	]
	for case in cases {
		input, grad, gamma, beta := ln_cudnn_fixture(case[0], case[1])!
		ln_cudnn_disable_env()
		cpu := layer_norm_backward_shape[f32](grad, input, gamma, beta, case[1], eps)!
		gpu := layer_norm_backward_cuda_f32(grad, input, gamma, eps, layer_norm_test_size(case[1]))!

		assert gpu.len == 3
		assert gpu[0].shape == input.shape
		assert gpu[1].shape == gamma.shape
		assert gpu[2].shape == beta.shape
		names := ['d_input', 'd_gamma', 'd_beta']
		for k in 0 .. 3 {
			assert_ln_slices_close(gpu[k].to_array(), cpu[k].to_array(), 1e-4, names[k])
		}
	}
}

fn test_layer_norm_backward_dispatch_selects_cuda_when_enabled() ! {
	if ln_cudnn_test_skip() {
		return
	}
	eps := 1e-5
	input, grad, gamma, beta := ln_cudnn_fixture([5, 4], [4])!
	ln_cudnn_disable_env()
	cpu := layer_norm_backward_shape[f32](grad, input, gamma, beta, [4], eps)!

	os.setenv('VTL_USE_CUDA', '1', true)
	os.setenv('VTL_CUDA_BACKWARD', '1', true)
	defer {
		ln_cudnn_disable_env()
	}
	assert layer_norm_cudnn_eligible()
	dispatched := layer_norm_backward_shape[f32](grad, input, gamma, beta, [4], eps)!
	names := ['d_input', 'd_gamma', 'd_beta']
	for k in 0 .. 3 {
		assert_ln_slices_close(dispatched[k].to_array(), cpu[k].to_array(), 1e-4, names[k])
	}
}

fn layer_norm_test_size(shape []int) int {
	mut size := 1
	for dim in shape {
		size *= dim
	}
	return size
}
