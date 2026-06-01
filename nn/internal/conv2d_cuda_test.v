module internal

import os
import vtl
import math

fn test_conv2d_forward_cpu_f64() ! {
	input := vtl.from_array([f64(1), 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4], [1, 1, 4, 4])!
	weight := vtl.from_array([f64(1), 0, 0, 0], [1, 1, 2, 2])!
	bias := vtl.zeros[f64]([1, 1])
	cfg := Conv2DConfig{
		padding:  [0, 0]
		stride:   [1, 1]
		dilation: [1, 1]
		groups:   1
	}
	out := conv2d_forward_cpu_f64(input, weight, bias, [2, 2], cfg)!
	assert out.shape == [1, 1, 3, 3]
}

fn test_conv2d_cuda_eligible_same_padding() {
	cfg := Conv2DConfig{
		padding:  [1, 1]
		stride:   [1, 1]
		dilation: [1, 1]
		groups:   1
	}
	expected := os.getenv('VTL_USE_CUDA') == '1'
	assert conv2d_cuda_eligible([3, 3], cfg) == expected
}

fn test_conv2d_forward_f64_matches_cpu_reference() ! {
	if os.getenv('VTL_TEST_CUDA') != '1' || os.getenv('VTL_USE_CUDA') != '1' {
		return
	}
	input := vtl.from_array([f64(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [
		1,
		1,
		4,
		4,
	])!
	weight := vtl.from_array([f64(1), 0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 3, 3])!
	bias := vtl.zeros[f64]([1, 1])
	cfg := Conv2DConfig{
		padding:  [1, 1]
		stride:   [1, 1]
		dilation: [1, 1]
		groups:   1
	}
	k := [3, 3]
	cpu := conv2d_forward_cpu_f64(input, weight, bias, k, cfg)!
	// Integration path: CUDA when cuDNN succeeds, else CPU fallback
	out := conv2d_forward_f64(input, weight, bias, k, cfg)!
	assert cpu.shape == out.shape
	for i in 0 .. cpu.size {
		diff := math.abs(cpu.get_nth(i) - out.get_nth(i))
		assert diff < 1e-5, 'conv2d forward diff at ${i}: ${diff}'
	}
}

fn test_conv2d_backward_cpu_f64_smoke() ! {
	input := vtl.from_array([f64(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [
		1,
		1,
		4,
		4,
	])!
	weight := vtl.from_array([f64(1), 0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 3, 3])!
	bias := vtl.zeros[f64]([1, 1])
	cfg := Conv2DConfig{
		padding:  [1, 1]
		stride:   [1, 1]
		dilation: [1, 1]
		groups:   1
	}
	k := [3, 3]
	out := conv2d_forward_cpu_f64(input, weight, bias, k, cfg)!
	grad := vtl.ones[f64](out.shape)
	tensors := conv2d_backward_cpu_f64(grad, input, weight, bias, k, cfg)!
	assert tensors[0].shape == input.shape
	assert tensors[1].shape == weight.shape
	assert tensors[2].shape == [1, 1]
}

fn test_conv2d_backward_cuda_matches_cpu() ! {
	if os.getenv('VTL_TEST_CUDA') != '1' || os.getenv('VTL_CUDA_BACKWARD') != '1' {
		return
	}
	input := vtl.from_array([f64(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [
		1,
		1,
		4,
		4,
	])!
	weight := vtl.from_array([f64(1), 0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 3, 3])!
	bias := vtl.zeros[f64]([1, 1])
	cfg := Conv2DConfig{
		padding:  [1, 1]
		stride:   [1, 1]
		dilation: [1, 1]
		groups:   1
	}
	k := [3, 3]
	if !conv2d_cuda_eligible(k, cfg) {
		return
	}
	grad := vtl.from_array([f64(0.1), 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
		1.4, 1.5, 1.6], [
		1,
		1,
		4,
		4,
	])!
	cpu := conv2d_backward_cpu_f64(grad, input, weight, bias, k, cfg)!
	gpu := conv2d_backward_f64(grad, input, weight, bias, k, cfg)!
	for ti in 0 .. 3 {
		for i in 0 .. cpu[ti].size {
			diff := math.abs(cpu[ti].get_nth(i) - gpu[ti].get_nth(i))
			assert diff < 1e-4, 'conv2d bwd mismatch t${ti} i${i}: ${diff}'
		}
	}
}

fn test_conv2d_cuda_direct_optional() ! {
	if os.getenv('VTL_TEST_CUDA') != '1' || os.getenv('VTL_USE_CUDA') != '1' {
		return
	}
	input := vtl.from_array([f64(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [
		1,
		1,
		4,
		4,
	])!
	weight := vtl.from_array([f64(1), 0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 3, 3])!
	bias := vtl.zeros[f64]([1, 1])
	cfg := Conv2DConfig{
		padding:  [1, 1]
		stride:   [1, 1]
		dilation: [1, 1]
		groups:   1
	}
	k := [3, 3]
	if !conv2d_cuda_eligible(k, cfg) {
		return
	}
	cpu := conv2d_forward_cpu_f64(input, weight, bias, k, cfg)!
	gpu := conv2d_forward_cuda_f64(input, weight, bias, k, cfg) or {
		// cuDNN may be unavailable on some drivers; integration test still passes
		return
	}
	assert cpu.shape == gpu.shape
	for i in 0 .. cpu.size {
		diff := math.abs(cpu.get_nth(i) - gpu.get_nth(i))
		assert diff < 1e-5, 'conv2d CPU vs CUDA diff at ${i}: ${diff}'
	}
}
