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
