// Compare f32, CPU INT8, one-shot CUDA INT8, and resident-weight CUDA INT8 Linear.
// Compile with `-d cuda -prod` from ~/.vmodules, then run the binary separately.
module main

import math
import os
import strconv
import time
import vtl
import vtl.la
import vtl.quantization
import vsl.cuda

const default_batch = 32
const default_in_features = 256
const default_out_features = 512
const warmup_runs = 3
const iterations = 24

struct BenchResult {
	ms       f64
	checksum f64
}

fn main() {
	batch := benchmark_dimension(1, default_batch)
	in_features := benchmark_dimension(2, default_in_features)
	out_features := benchmark_dimension(3, default_out_features)
	if batch > max_int / in_features || out_features > max_int / in_features
		|| batch * in_features > max_int / out_features {
		panic('benchmark dimensions exceed supported integer sizes')
	}
	input := make_input(batch, in_features) or { panic(err) }
	weights := make_weights(in_features, out_features) or { panic(err) }
	bias := make_bias(out_features) or { panic(err) }
	dev := cuda.get_default_device() or { panic(err) }

	mut input_observer := quantization.MinMaxObserver{}
	input_observer.observe_f32(input) or { panic(err) }
	input_params := input_observer.params(-128, 127) or { panic(err) }
	mut weight_observer := quantization.new_per_channel_min_max_observer(0)
	weight_observer.observe_f32(weights) or { panic(err) }
	weight_params := weight_observer.params(-128, 127) or { panic(err) }
	quantized_weights := quantization.quantize_per_channel_f32(weights, weight_params) or {
		panic(err)
	}
	cpu_layer := &quantization.QuantizedLinearLayer{
		weights:       quantized_weights
		weight_params: weight_params
		input_params:  input_params
		bias:          bias
	}
	mut resident_cuda_layer := quantization.upload_quantized_linear_layer_cuda(dev, cpu_layer) or {
		panic(err)
	}
	defer {
		resident_cuda_layer.release() or { panic(err) }
	}

	reference := linear_f32(input, weights, bias, out_features) or { panic(err) }
	cpu := quantization.linear_int8_f32(input, quantized_weights, input_params, weight_params,
		bias) or {
		panic(err)
	}
	gpu := quantization.linear_int8_f32_cuda(dev, input, quantized_weights, input_params,
		weight_params, bias) or {
		panic(err)
	}
	resident_gpu := resident_cuda_layer.forward(input) or { panic(err) }
	cpu_error := max_abs_difference(reference, cpu)
	gpu_error := max_abs_difference(reference, gpu)
	resident_gpu_error := max_abs_difference(reference, resident_gpu)
	assert math.is_finite(cpu_error) && math.is_finite(gpu_error)
	assert math.is_finite(resident_gpu_error)

	for _ in 0 .. warmup_runs {
		_ = linear_f32(input, weights, bias, out_features) or { panic(err) }
		_ = quantization.linear_int8_f32(input, quantized_weights, input_params,
			weight_params, bias) or {
			panic(err)
		}
		_ = quantization.linear_int8_f32_cuda(dev, input, quantized_weights, input_params,
			weight_params, bias) or {
			panic(err)
		}
		_ = resident_cuda_layer.forward(input) or { panic(err) }
	}
	float32 := time_float32(input, weights, bias, out_features) or { panic(err) }
	cpu_int8 := time_int8(input, quantized_weights, input_params, weight_params, bias) or {
		panic(err)
	}
	gpu_int8 := time_cuda(dev, input, quantized_weights, input_params, weight_params, bias) or {
		panic(err)
	}
	resident_gpu_int8 := time_cuda_resident(resident_cuda_layer, input) or { panic(err) }

	println('device_id: ${dev.device_id}')
	println('shape: [${batch}, ${in_features}] x [${out_features}, ${in_features}]')
	println('warmups: ${warmup_runs}; iterations: ${iterations}')
	println('f32_ms: ${float32.ms:.4f}; cpu_int8_ms: ${cpu_int8.ms:.4f}; cuda_int8_ms: ${gpu_int8.ms:.4f}; cuda_resident_int8_ms: ${resident_gpu_int8.ms:.4f}')
	println('speedup_vs_f32: cpu=${float32.ms / cpu_int8.ms:.3f}x; cuda=${float32.ms / gpu_int8.ms:.3f}x; cuda_resident=${float32.ms / resident_gpu_int8.ms:.3f}x')
	println('max_abs_error_vs_f32: cpu=${cpu_error:.6f}; cuda=${gpu_error:.6f}; cuda_resident=${resident_gpu_error:.6f}')
	println('checksums: f32=${float32.checksum:.6f}; cpu_int8=${cpu_int8.checksum:.6f}; cuda_int8=${gpu_int8.checksum:.6f}; cuda_resident=${resident_gpu_int8.checksum:.6f}')
}

fn benchmark_dimension(index int, fallback int) int {
	if os.args.len <= index {
		return fallback
	}
	value := strconv.atoi(os.args[index]) or {
		panic('dimension argument ${index} must be a positive integer')
	}
	if value <= 0 {
		panic('dimension argument ${index} must be a positive integer')
	}
	return value
}

fn make_input(batch int, in_features int) !&vtl.Tensor[f32] {
	mut values := []f32{len: batch * in_features}
	for index in 0 .. values.len {
		values[index] = f32((index * 17) % 101 - 50) / 50.0
	}
	return vtl.from_array[f32](values, [batch, in_features])
}

fn make_weights(in_features int, out_features int) !&vtl.Tensor[f32] {
	mut values := []f32{len: out_features * in_features}
	for index in 0 .. values.len {
		values[index] = f32((index * 13) % 97 - 48) / 128.0
	}
	return vtl.from_array[f32](values, [out_features, in_features])
}

fn make_bias(out_features int) !&vtl.Tensor[f32] {
	mut values := []f32{len: out_features}
	for index in 0 .. values.len {
		values[index] = f32(index % 17 - 8) / 32.0
	}
	return vtl.from_array[f32](values, [1, out_features])
}

fn linear_f32(input &vtl.Tensor[f32], weights &vtl.Tensor[f32], bias &vtl.Tensor[f32], out_features int) !&vtl.Tensor[f32] {
	mut output := la.matmul[f32](input, weights.t()!)!
	bias_values := bias.to_array()
	for index in 0 .. output.size {
		output.data.data[index] += bias_values[index % out_features]
	}
	return output
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

fn time_float32(input &vtl.Tensor[f32], weights &vtl.Tensor[f32], bias &vtl.Tensor[f32], out_features int) !BenchResult {
	mut checksum := 0.0
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		result := linear_f32(input, weights, bias, out_features)!
		checksum += f64(result.get_nth(0))
	}
	return BenchResult{
		ms:       f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
		checksum: checksum
	}
}

fn time_int8(input &vtl.Tensor[f32], weights &vtl.Tensor[i8], input_params quantization.QuantizationParams, weight_params quantization.PerChannelParams, bias &vtl.Tensor[f32]) !BenchResult {
	mut checksum := 0.0
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		result := quantization.linear_int8_f32(input, weights, input_params, weight_params, bias)!
		checksum += f64(result.get_nth(0))
	}
	return BenchResult{
		ms:       f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
		checksum: checksum
	}
}

fn time_cuda(dev &cuda.CudaDevice, input &vtl.Tensor[f32], weights &vtl.Tensor[i8], input_params quantization.QuantizationParams, weight_params quantization.PerChannelParams, bias &vtl.Tensor[f32]) !BenchResult {
	mut checksum := 0.0
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		result := quantization.linear_int8_f32_cuda(dev, input, weights, input_params,
			weight_params, bias)!
		checksum += f64(result.get_nth(0))
	}
	return BenchResult{
		ms:       f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
		checksum: checksum
	}
}

fn time_cuda_resident(layer &quantization.CudaQuantizedLinearLayer, input &vtl.Tensor[f32]) !BenchResult {
	mut checksum := 0.0
	started := time.sys_mono_now()
	for _ in 0 .. iterations {
		result := layer.forward(input)!
		checksum += f64(result.get_nth(0))
	}
	return BenchResult{
		ms:       f64(time.sys_mono_now() - started) / f64(iterations) / 1_000_000.0
		checksum: checksum
	}
}
