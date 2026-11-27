// Compare CPU and CUDA INT8 Conv2D inference with shared deterministic inputs.
// Compile with `-d cuda -prod` from ~/.vmodules, then run separately.
module main

import math
import time
import vtl
import vtl.quantization
import vsl.cuda

const batch = 1
const in_channels = 8
const out_channels = 16
const height = 32
const width = 32
const kernel_size = [3, 3]
const warmup_runs = 3
const iterations = 24

struct BenchResult {
	ms       f64
	checksum f64
}

fn main() {
	input := make_input() or { panic(err) }
	weights := make_weights() or { panic(err) }
	bias := make_bias() or { panic(err) }
	dev := cuda.get_default_device() or { panic(err) }
	config := quantization.QuantizedConv2DConfig{
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

	cpu := quantization.conv2d_int8_f32(input, quantized_weights, input_params,
		weight_params, bias, kernel_size, config) or {
		panic(err)
	}
	gpu := quantization.conv2d_int8_f32_cuda(dev, input, quantized_weights, input_params,
		weight_params, bias, kernel_size, config) or {
		panic(err)
	}
	max_error := max_abs_difference(cpu, gpu)
	assert math.is_finite(max_error)

	for _ in 0 .. warmup_runs {
		_ = quantization.conv2d_int8_f32(input, quantized_weights, input_params,
			weight_params, bias, kernel_size, config) or {
			panic(err)
		}
		_ = quantization.conv2d_int8_f32_cuda(dev, input, quantized_weights, input_params,
			weight_params, bias, kernel_size, config) or {
			panic(err)
		}
	}
	cpu_result := time_cpu(input, quantized_weights, input_params, weight_params, bias,
		config) or {
		panic(err)
	}
	cuda_result := time_cuda(dev, input, quantized_weights, input_params, weight_params,
		bias, config) or {
		panic(err)
	}
	weight_payload_bytes := quantized_weights.size * sizeof(i8)
	metadata_bytes := weight_params.scales.len * (sizeof(f64) + sizeof(int))

	println('device_id: ${dev.device_id}')
	println('shape: [${batch}, ${in_channels}, ${height}, ${width}] x [${out_channels}, ${in_channels}, ${kernel_size[0]}, ${kernel_size[1]}]')
	println('warmups: ${warmup_runs}; iterations: ${iterations}')
	println('cpu_int8_ms: ${cpu_result.ms:.4f}; cuda_int8_ms: ${cuda_result.ms:.4f}; speedup: ${cpu_result.ms / cuda_result.ms:.3f}x')
	println('max_abs_error_vs_cpu: ${max_error:.6f}; checksums: ${cpu_result.checksum:.6f}, ${cuda_result.checksum:.6f}')
	println('weight_payload_bytes: int8=${weight_payload_bytes}; per_channel_metadata=${metadata_bytes} (bias and V object overhead excluded)')
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

fn time_cpu(input &vtl.Tensor[f32], weights &vtl.Tensor[i8], input_params quantization.QuantizationParams, weight_params quantization.PerChannelParams, bias &vtl.Tensor[f32], config quantization.QuantizedConv2DConfig) !BenchResult {
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

fn time_cuda(dev &cuda.CudaDevice, input &vtl.Tensor[f32], weights &vtl.Tensor[i8], input_params quantization.QuantizationParams, weight_params quantization.PerChannelParams, bias &vtl.Tensor[f32], config quantization.QuantizedConv2DConfig) !BenchResult {
	mut checksum := 0.0
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		result := quantization.conv2d_int8_f32_cuda(dev, input, weights, input_params,
			weight_params, bias, kernel_size, config)!
		checksum += f64(result.get_nth(0))
	}
	return BenchResult{
		ms:       f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
		checksum: checksum
	}
}
