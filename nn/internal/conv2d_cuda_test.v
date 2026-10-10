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

fn test_conv2d_forward_cpu_f64_contiguous_fast_path_matches_reference() ! {
	mut input_values := []f64{len: 2 * 4 * 5 * 6}
	for i in 0 .. input_values.len {
		input_values[i] = f64(i % 29 - 14) / 13.0
	}
	mut weight_values := []f64{len: 6 * 2 * 2 * 3}
	for i in 0 .. weight_values.len {
		weight_values[i] = f64(i % 17 - 8) / 11.0
	}
	input := vtl.from_array(input_values, [2, 4, 5, 6])!
	weight := vtl.from_array(weight_values, [6, 2, 2, 3])!
	bias := vtl.from_array([f64(0.1), 0.2, 0.3, 0.4, 0.5, 0.6], [1, 6])!
	config := Conv2DConfig{
		padding:  [1, 1]
		stride:   [2, 2]
		dilation: [2, 1]
		groups:   2
	}
	kernel_size := [2, 3]
	optimized := conv2d_forward_cpu_f64(input, weight, bias, kernel_size, config)!
	reference := conv2d_forward_cpu_f64_reference(input, weight, bias, kernel_size, config)!
	assert optimized.shape == reference.shape
	for i in 0 .. optimized.size {
		assert math.abs(optimized.get_nth(i) - reference.get_nth(i)) < 1e-12
	}
}

fn test_conv2d_forward_cpu_f32_contiguous_fast_path_matches_reference() ! {
	mut input_values := []f32{len: 2 * 4 * 5 * 6}
	for i in 0 .. input_values.len {
		input_values[i] = f32(i % 29 - 14) / 13.0
	}
	mut weight_values := []f32{len: 6 * 2 * 2 * 3}
	for i in 0 .. weight_values.len {
		weight_values[i] = f32(i % 17 - 8) / 11.0
	}
	input := vtl.from_array(input_values, [2, 4, 5, 6])!
	weight := vtl.from_array(weight_values, [6, 2, 2, 3])!
	bias := vtl.from_array([f32(0.1), 0.2, 0.3, 0.4, 0.5, 0.6], [1, 6])!
	config := Conv2DConfig{
		padding:  [1, 1]
		stride:   [2, 2]
		dilation: [2, 1]
		groups:   2
	}
	kernel_size := [2, 3]
	optimized := conv2d_forward_cpu_f32(input, weight, bias, kernel_size, config)!
	reference := conv2d_forward_cpu_f32_reference(input, weight, bias, kernel_size, config)!
	assert optimized.shape == reference.shape
	for i in 0 .. optimized.size {
		assert math.abs(f64(optimized.get_nth(i) - reference.get_nth(i))) < 1e-5
	}
}

fn test_conv2d_forward_cpu_f32_supports_reversed_input_view() ! {
	mut input := vtl.from_array([f32(1.0), 2.0], [1, 1, 1, 2])!
	input.strides[3] = -1
	weight := vtl.from_array([f32(1.0)], [1, 1, 1, 1])!
	bias := vtl.from_array([f32(0.0)], [1, 1])!
	output := conv2d_forward_cpu_f32(input, weight, bias, [1, 1], Conv2DConfig{})!
	assert output.to_array() == [f32(2.0), 1.0]
}

fn test_conv2d_forward_cpu_f64_rejects_invalid_tensor_metadata() {
	mut input := vtl.from_array([f64(1.0)], [1, 1, 1, 1]) or { panic(err) }
	input.shape = [1, 1]
	weight := vtl.from_array([f64(1.0)], [1, 1, 1, 1]) or { panic(err) }
	bias := vtl.from_array([f64(0.0)], [1, 1]) or { panic(err) }
	config := Conv2DConfig{}
	if _ := conv2d_forward_cpu_f64(input, weight, bias, [1, 1], config) {
		assert false, 'expected invalid input rank to be rejected'
	} else {
		assert err.msg().contains('input shape and strides')
	}
}

fn test_conv2d_forward_cpu_f64_rejects_out_of_bounds_strides() {
	mut input := vtl.from_array([f64(1.0), 2.0], [1, 1, 1, 2]) or { panic(err) }
	input.strides[3] = 3
	weight := vtl.from_array([f64(1.0)], [1, 1, 1, 1]) or { panic(err) }
	bias := vtl.from_array([f64(0.0)], [1, 1]) or { panic(err) }
	if _ := conv2d_forward_cpu_f64(input, weight, bias, [1, 1], Conv2DConfig{}) {
		assert false, 'expected out-of-bounds strides to be rejected'
	} else {
		assert err.msg().contains('storage is smaller')
	}
}

fn test_conv2d_forward_cpu_f64_supports_reversed_input_view() ! {
	mut input := vtl.from_array([f64(1.0), 2.0], [1, 1, 1, 2])!
	input.strides[3] = -1
	weight := vtl.from_array([f64(1.0)], [1, 1, 1, 1])!
	bias := vtl.from_array([f64(0.0)], [1, 1])!
	output := conv2d_forward_cpu_f64(input, weight, bias, [1, 1], Conv2DConfig{})!
	assert output.to_array() == [2.0, 1.0]
}

fn test_conv2d_forward_cpu_f64_rejects_overflowing_negative_stride() {
	mut input := vtl.from_array([f64(1.0), 2.0, 3.0], [1, 1, 1, 3]) or { panic(err) }
	input.strides[3] = -max_int
	weight := vtl.from_array([f64(1.0)], [1, 1, 1, 1]) or { panic(err) }
	bias := vtl.from_array([f64(0.0)], [1, 1]) or { panic(err) }
	if _ := conv2d_forward_cpu_f64(input, weight, bias, [1, 1], Conv2DConfig{}) {
		assert false, 'expected overflowing stride to be rejected'
	} else {
		assert err.msg().contains('strides are invalid or too large')
	}
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
