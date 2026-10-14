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
}
