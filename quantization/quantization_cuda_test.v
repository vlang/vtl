module quantization

import math
import os
import vtl
import vsl.cuda

fn test_linear_int8_f32_cuda_matches_cpu_with_affine_zero_points() ! {
	if os.getenv('VTL_TEST_CUDA') != '1' {
		return
	}
	dev := cuda.get_default_device()!
	input := vtl.from_array[f32]([0.3, -1.2, 2.4, 1.1, 0.0, -0.5], [2, 3])!
	weights := vtl.from_array[i8]([7, -8, 11, -10, 12, 5], [2, 3])!
	bias := vtl.from_1d[f32]([0.25, -0.75])!
	input_params := new_params(0.05, 5, -128, 127)!
	weight_params := new_per_channel_params([0.2, 0.3], [7, -4], 0, -128, 127)!
	cpu := linear_int8_f32(input, weights, input_params, weight_params, bias)!
	gpu := linear_int8_f32_cuda(dev, input, weights, input_params, weight_params, bias)!
	cpu_values := cpu.to_array()
	gpu_values := gpu.to_array()
	assert gpu.shape == cpu.shape
	for i in 0 .. cpu_values.len {
		assert math.abs(gpu_values[i] - cpu_values[i]) < 1e-6, 'CUDA INT8 Linear mismatch at ${i}: got ${gpu_values[i]}, want ${cpu_values[i]}'
	}
}

fn test_conv2d_int8_f32_cuda_matches_cpu_with_padding_and_affine_zero_points() ! {
	if os.getenv('VTL_TEST_CUDA') != '1' {
		return
	}
	dev := cuda.get_default_device()!
	input := vtl.from_array[f32]([1, 2, 3, 4, 5, 6], [1, 1, 2, 3])!
	weights := vtl.from_array[i8]([55, 105, -55, 20], [2, 1, 1, 2])!
	bias := vtl.from_array[f32]([0.25, -0.5], [1, 2])!
	input_params := new_params(0.1, 3, -128, 127)!
	weight_params := new_per_channel_params([0.01, 0.02], [5, -5], 0, -128, 127)!
	config := QuantizedConv2DConfig{
		padding:  [0, 1]
		stride:   [1, 2]
		dilation: [1, 2]
	}
	cpu := conv2d_int8_f32(input, weights, input_params, weight_params, bias, [1, 2], config)!
	gpu := conv2d_int8_f32_cuda(dev, input, weights, input_params, weight_params, bias, [
		1,
		2,
	],
		config)!
	cpu_values := cpu.to_array()
	gpu_values := gpu.to_array()
	assert gpu.shape == cpu.shape
	for i in 0 .. cpu_values.len {
		assert math.abs(gpu_values[i] - cpu_values[i]) < 1e-6, 'CUDA INT8 Conv2D mismatch at ${i}: got ${gpu_values[i]}, want ${cpu_values[i]}'
	}
}

fn test_conv2d_int8_f32_cuda_handles_batches_and_groups() ! {
	if os.getenv('VTL_TEST_CUDA') != '1' {
		return
	}
	dev := cuda.get_default_device()!
	input := vtl.from_array[f32]([1, 2, 3, 4], [2, 2, 1, 1])!
	weights := vtl.from_array[i8]([10, 20], [2, 1, 1, 1])!
	bias := vtl.from_array[f32]([0.25, -0.5], [1, 2])!
	input_params := new_params(0.1, 3, -128, 127)!
	weight_params := new_per_channel_params([0.1, 0.2], [2, -1], 0, -128, 127)!
	config := QuantizedConv2DConfig{
		groups: 2
	}
	cpu := conv2d_int8_f32(input, weights, input_params, weight_params, bias, [1, 1], config)!
	gpu := conv2d_int8_f32_cuda(dev, input, weights, input_params, weight_params, bias, [
		1,
		1,
	],
		config)!
	cpu_values := cpu.to_array()
	gpu_values := gpu.to_array()
	assert gpu.shape == cpu.shape
	for i in 0 .. cpu_values.len {
		assert math.abs(gpu_values[i] - cpu_values[i]) < 1e-6, 'batched grouped CUDA INT8 Conv2D mismatch at ${i}: got ${gpu_values[i]}, want ${cpu_values[i]}'
	}
}

$if cuda ? {
	fn test_cuda_quantized_linear_layer_reuses_device_weights() ! {
		if os.getenv('VTL_TEST_CUDA') != '1' {
			return
		}
		dev := cuda.get_default_device()!
		input := vtl.from_array[f32]([0.3, -1.2, 2.4, 1.1, 0.0, -0.5], [2, 3])!
		weights := vtl.from_array[f32]([0.7, -0.8, 1.1, -1.0, 1.2, 0.5], [2, 3])!
		bias := vtl.from_1d[f32]([0.25, -0.75])!
		input_params := new_params(0.05, 5, -128, 127)!
		cpu_layer := quantize_linear_layer_f32(weights, bias, input_params, -128, 127)!
		mut cuda_layer := upload_quantized_linear_layer_cuda(dev, cpu_layer)!
		defer {
			cuda_layer.release() or { panic(err) }
		}
		cpu := cpu_layer.forward(input)!
		mut source_weights := cpu_layer.weights
		source_weights.data.data[0] = source_weights.data.data[0] + 1
		for _ in 0 .. 2 {
			gpu := cuda_layer.forward(input)!
			cpu_values := cpu.to_array()
			gpu_values := gpu.to_array()
			assert gpu.shape == cpu.shape
			for i in 0 .. cpu_values.len {
				assert math.abs(gpu_values[i] - cpu_values[i]) < 1e-6, 'retained CUDA INT8 Linear mismatch at ${i}'
			}
		}
	}
}
