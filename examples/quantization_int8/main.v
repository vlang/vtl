module main

import vtl
import vtl.quantization

fn main() {
	mut observer := quantization.MinMaxObserver{}
	calibration := vtl.from_1d([-1.0, 0.0, 0.5, 1.0]) or { panic(err) }
	observer.observe_f64(calibration) or { panic(err) }
	params := observer.params(-128, 127) or { panic(err) }
	encoded := quantization.quantize_per_tensor_f64(calibration, params) or { panic(err) }
	reconstructed := quantization.dequantize_per_tensor_f64(encoded, params) or { panic(err) }
	println('scale: ${params.scale}, zero point: ${params.zero_point}')
	println('INT8 values: ${encoded.to_array()}')
	println('reconstructed: ${reconstructed.to_array()}')

	input := vtl.from_array([f32(1), 2, -1, 0.5], [2, 2]) or { panic(err) }
	input_params := quantization.new_params(0.1, 0, -128, 127) or { panic(err) }
	weights := vtl.from_array([f32(0.5), -1, 2, 0.24], [2, 2]) or { panic(err) }
	weight_params := quantization.new_per_channel_params([0.01, 0.02], [0, 0], 0,
		-128, 127) or { panic(err) }
	quantized_weights := quantization.quantize_per_channel_f32(weights, weight_params) or {
		panic(err)
	}
	bias := vtl.from_array([f32(0.25), -0.5], [1, 2]) or { panic(err) }
	output := quantization.linear_int8_f32(input, quantized_weights, input_params,
		weight_params, bias) or {
		panic(err)
	}
	println('INT8 linear output: ${output.to_array()}')
}
