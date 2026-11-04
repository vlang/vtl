module internal

import math
import os
import vtl

// Gradient check: cuDNN MaxPool2D backward against the CPU reference.
// Runs only with `-d cuda` and VTL_TEST_CUDA=1 (same skip pattern as conv2d_cuda_test.v).

// maxpool2d_cuda_test_input returns values that are distinct under a 0.1 grid, so no
// pooling window holds two equal maxima and the GPU path is eligible.
fn maxpool2d_cuda_test_input(shape []int) !&vtl.Tensor[f64] {
	mut values := []f64{len: shape[0] * shape[1] * shape[2] * shape[3]}
	for i in 0 .. values.len {
		values[i] = f64((i * 7919) % 1000) / 10.0 - 50.0
	}
	return vtl.from_array(values, shape)!
}

fn maxpool2d_cuda_test_grad(shape []int) !&vtl.Tensor[f64] {
	mut values := []f64{len: shape[0] * shape[1] * shape[2] * shape[3]}
	for i in 0 .. values.len {
		values[i] = f64(i % 9 - 4) * 0.25 + 0.05
	}
	return vtl.from_array(values, shape)!
}

fn assert_maxpool_slices_close(got []f64, want []f64, tol f64, label string) {
	assert got.len == want.len, '${label}: length ${got.len} != ${want.len}'
	for i in 0 .. got.len {
		assert math.abs(got[i] - want[i]) < tol, '${label}[${i}]: got ${got[i]}, want ${want[i]}'
	}
}

fn test_maxpool2d_backward_cuda_matches_cpu_reference() ! {
	$if !cuda {
		return
	}
	if os.getenv('VTL_TEST_CUDA') != '1' {
		return
	}
	// Non-overlapping 2x2 windows, then overlapping 3x3 windows with stride 2.
	configs := [
		[6, 6, 2, 2],
		[7, 7, 3, 2],
	]
	for cfg in configs {
		input_shape := [2, 3, cfg[0], cfg[1]]
		kernel := [cfg[2], cfg[2]]
		stride := [cfg[3], cfg[3]]
		padding := [0, 0]
		input := maxpool2d_cuda_test_input(input_shape)!
		max_indices, output := maxpool2d[f64](input, kernel, padding, stride)
		grad := maxpool2d_cuda_test_grad(output.shape)!

		cpu := maxpool2d_backward[f64](input_shape, max_indices, grad)!
		gpu := maxpool2d_backward_cuda_f64(input_shape, max_indices, input, grad, kernel, stride)!

		assert gpu.shape == input_shape
		assert_maxpool_slices_close(gpu.to_array(), cpu.to_array(), 1e-9, 'd_input ${cfg}')
	}
}

fn test_maxpool2d_backward_dispatch_uses_cuda_only_when_eligible() ! {
	$if !cuda {
		return
	}
	if os.getenv('VTL_TEST_CUDA') != '1' {
		return
	}
	kernel := [2, 2]
	stride := [2, 2]
	padding := [0, 0]

	os.setenv('VTL_USE_CUDA', '1', true)
	os.setenv('VTL_CUDA_BACKWARD', '1', true)
	defer {
		os.unsetenv('VTL_USE_CUDA')
		os.unsetenv('VTL_CUDA_BACKWARD')
	}

	// Distinct values: the dispatch must choose cuDNN.
	distinct := maxpool2d_cuda_test_input([2, 3, 6, 6])!
	distinct_indices, distinct_out := maxpool2d[f64](distinct, kernel, padding, stride)
	distinct_grad := maxpool2d_cuda_test_grad(distinct_out.shape)!
	assert maxpool2d_cudnn_eligible(distinct, distinct_indices, distinct_grad, kernel, stride, padding)
	dispatched := maxpool2d_backward_with_input[f64](distinct.shape, distinct_indices, distinct,
		distinct_grad, kernel, stride, padding)!
	cpu := maxpool2d_backward[f64](distinct.shape, distinct_indices, distinct_grad)!
	assert_maxpool_slices_close(dispatched.to_array(), cpu.to_array(), 1e-9, 'dispatched distinct')

	// All-equal values: every window is tied, so the dispatch must fall back to the CPU rule.
	tied := vtl.zeros[f64]([1, 1, 4, 4])
	tied_indices, tied_out := maxpool2d[f64](tied, kernel, padding, stride)
	tied_grad := maxpool2d_cuda_test_grad(tied_out.shape)!
	assert !maxpool2d_cudnn_eligible(tied, tied_indices, tied_grad, kernel, stride, padding)
	tied_dispatched := maxpool2d_backward_with_input[f64](tied.shape, tied_indices, tied,
		tied_grad, kernel, stride, padding)!
	tied_cpu := maxpool2d_backward[f64](tied.shape, tied_indices, tied_grad)!
	assert_maxpool_slices_close(tied_dispatched.to_array(), tied_cpu.to_array(), 1e-12, 'dispatched tied')
}
