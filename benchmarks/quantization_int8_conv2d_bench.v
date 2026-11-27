// Compare end-to-end f32 Conv2D with quantized CPU inference.
// Compile with `-prod` from ~/.vmodules, then run the binary separately.
module main

import math
import time
import vtl
import vtl.nn.internal as nn_internal
import vtl.quantization

const batch = 1
const in_channels = 8
const out_channels = 16
const height = 32
const width = 32
const kernel_size = [3, 3]
const warmup_runs = 1
const iterations = 3

struct BenchResult {
	ms       f64
	checksum f64
}

fn main() {
	input := make_input() or { panic(err) }
	weights := make_weights() or { panic(err) }
	bias := make_bias() or { panic(err) }
	config := nn_internal.Conv2DConfig{
		padding: [1, 1]
	}
	quantized_config := quantization.QuantizedConv2DConfig{
		padding: [1, 1]
	}

	mut input_observer := quantization.MinMaxObserver{}
	input_observer.observe_f32(input) or { panic(err) }
	input_params := input_observer.params(-128, 127) or { panic(err) }
	mut weight_observer := quantization.new_per_channel_min_max_observer(0)
	weight_observer.observe_f32(weights) or { panic(err) }
	weight_params := weight_observer.params(-128, 127) or { panic(err) }
	quantized_weights := quantization.quantize_per_channel_f32(weights, weight_params) or {
		panic(err)
	}

	reference := nn_internal.conv2d_forward_f32(input, weights, bias, kernel_size,
		config) or {
		panic(err)
	}
	quantized := quantization.conv2d_int8_f32(input, quantized_weights, input_params,
		weight_params, bias, kernel_size, quantized_config) or {
		panic(err)
	}
	max_error := max_abs_difference(reference, quantized)
	assert math.is_finite(max_error)

	for _ in 0 .. warmup_runs {
		_ = nn_internal.conv2d_forward_f32(input, weights, bias, kernel_size, config) or {
			panic(err)
		}
		_ = quantization.conv2d_int8_f32(input, quantized_weights, input_params,
			weight_params, bias, kernel_size, quantized_config) or {
			panic(err)
		}
	}
	float32 := time_float32(input, weights, bias, config) or { panic(err) }
	int8 := time_int8(input, quantized_weights, input_params, weight_params, bias,
		quantized_config) or {
		panic(err)
	}

	f32_weight_bytes := weights.size * sizeof(f32)
	int8_weight_bytes := quantized_weights.size * sizeof(i8)
	metadata_bytes := weight_params.scales.len * (sizeof(f64) + sizeof(int))
	effective_int8_weight_bytes := int8_weight_bytes + metadata_bytes
	compression_ratio := f64(f32_weight_bytes) / f64(effective_int8_weight_bytes)

	println('shape: [${batch}, ${in_channels}, ${height}, ${width}] x [${out_channels}, ${in_channels}, ${kernel_size[0]}, ${kernel_size[1]}]')
	println('warmups: ${warmup_runs}; iterations: ${iterations}')
	println('float32_conv2d_ms: ${float32.ms:.4f}; int8_conv2d_ms: ${int8.ms:.4f}; speedup: ${float32.ms / int8.ms:.3f}x')
	println('max_abs_error: ${max_error:.6f}; checksums: ${float32.checksum:.6f}, ${int8.checksum:.6f}')
	println('weight_payload_bytes: f32=${f32_weight_bytes}; int8=${int8_weight_bytes}; per_channel_metadata=${metadata_bytes}')
	println('effective_weight_compression_ratio: ${compression_ratio:.3f}x (bias excluded; V object overhead excluded)')
}

fn make_input() !&vtl.Tensor[f32] {
	mut values := []f32{len: batch * in_channels * height * width}
	for index in 0 .. values.len {
		values[index] = f32((index * 17) % 101 - 50) / 50.0
	}
	return vtl.from_array[f32](values, [batch, in_channels, height, width])
}

fn make_weights() !&vtl.Tensor[f32] {
	mut values := []f32{len: out_channels * in_channels * kernel_size[0] * kernel_size[1]}
	for index in 0 .. values.len {
		values[index] = f32((index * 13) % 47 - 23) / 32.0
	}
	return vtl.from_array[f32](values, [
		out_channels,
		in_channels,
		kernel_size[0],
		kernel_size[1],
	])
}

fn make_bias() !&vtl.Tensor[f32] {
	mut values := []f32{len: out_channels}
	for index in 0 .. values.len {
		values[index] = f32(index % 17 - 8) / 32.0
	}
	return vtl.from_array[f32](values, [1, out_channels])
}

fn max_abs_difference(left &vtl.Tensor[f32], right &vtl.Tensor[f32]) f64 {
	mut maximum := 0.0
	for index in 0 .. left.size {
		difference := math.abs(f64(left.get_nth(index) - right.get_nth(index)))
		if difference > maximum {
			maximum = difference
		}
	}
	return maximum
}

fn time_float32(input &vtl.Tensor[f32], weights &vtl.Tensor[f32], bias &vtl.Tensor[f32], config nn_internal.Conv2DConfig) !BenchResult {
	mut checksum := 0.0
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		result := nn_internal.conv2d_forward_f32(input, weights, bias, kernel_size, config)!
		checksum += f64(result.get_nth(0))
	}
	return BenchResult{
		ms:       f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
		checksum: checksum
	}
}

fn time_int8(input &vtl.Tensor[f32], weights &vtl.Tensor[i8], input_params quantization.QuantizationParams, weight_params quantization.PerChannelParams, bias &vtl.Tensor[f32], config quantization.QuantizedConv2DConfig) !BenchResult {
	mut checksum := 0.0
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		result := quantization.conv2d_int8_f32(input, weights, input_params, weight_params,
			bias, kernel_size, config)!
		checksum += f64(result.get_nth(0))
	}
	return BenchResult{
		ms:       f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
		checksum: checksum
	}
}
