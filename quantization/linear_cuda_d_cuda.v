module quantization

import math
import vtl
import vsl.cuda
import vsl.cuda.compute as cuda_compute

// linear_int8_f32_cuda runs affine-INT8 Linear inference with cuBLAS.
// The result is a CPU-resident f32 tensor; the caller owns the CUDA device.
pub fn linear_int8_f32_cuda(dev &cuda.CudaDevice, input &vtl.Tensor[f32], weights &vtl.Tensor[i8], input_params QuantizationParams, weight_params PerChannelParams, bias &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	return linear_int8_f32_cuda_impl(dev, input, weights, input_params, weight_params, bias,
		none, none)
}

fn linear_int8_f32_cuda_impl(dev &cuda.CudaDevice, input &vtl.Tensor[f32], weights &vtl.Tensor[i8], input_params QuantizationParams, weight_params PerChannelParams, bias &vtl.Tensor[f32], preuploaded_b ?&cuda_compute.CudaI8Matrix, cached_weight_sums ?[]i64) !&vtl.Tensor[f32] {
	if input.rank() == 0 {
		return error('quantized linear input must have at least one dimension')
	}
	if weights.rank() != 2 {
		return error('quantized linear weights must have shape [out_features, in_features]')
	}
	in_features := input.shape[input.rank() - 1]
	out_features := weights.shape[0]
	if weights.shape[1] != in_features {
		return error('quantized linear input has ${in_features} features, weights expect ${weights.shape[1]}')
	}
	if out_features == 0 {
		return error('quantized linear weights must contain at least one output feature')
	}
	bias_is_vector := bias.rank() == 1 && bias.shape[0] == out_features
	bias_is_row := bias.rank() == 2 && bias.shape[0] == 1 && bias.shape[1] == out_features
	if !bias_is_vector && !bias_is_row {
		return error('quantized linear bias must have shape [out_features] or [1, out_features]')
	}
	weight_axis := normalize_axis(weight_params.axis, weights.rank())!
	if weight_axis != 0 {
		return error('quantized linear weight parameters must use output-feature axis 0')
	}
	validate_channel_params(weight_params, out_features)!
	validate_params(input_params)!

	mut rows := 1
	for dimension in input.shape[..input.shape.len - 1] {
		if dimension < 0 || (dimension > 0 && rows > max_int / dimension) {
			return error('quantized linear input shape exceeds supported integer sizes')
		}
		rows *= dimension
	}
	if rows > 0 && out_features > max_int / rows {
		return error('quantized linear output shape exceeds supported integer sizes')
	}

	quantized_input := quantize_per_tensor_f32(input, input_params)!
	input_values := quantized_input.to_array()
	weight_values := if cached_weight_sums == none { weights.to_array() } else { []i8{} }
	bias_values := bias.to_array()
	if input_values.len != input.size || (cached_weight_sums == none && weight_values.len != weights.size)
		|| bias_values.len != out_features {
		return error('quantized linear tensor storage does not match its shape')
	}

	mut weight_sums := []i64{}
	if cached_weight_sums == none {
		weight_sums = []i64{len: out_features}
		for output_feature in 0 .. out_features {
			weight_start := output_feature * in_features
			for feature in 0 .. in_features {
				weight := weight_values[weight_start + feature]
				if weight < weight_params.qmin || weight > weight_params.qmax {
					return error('quantized weight at logical index ${weight_start + feature} is outside [${weight_params.qmin}, ${weight_params.qmax}]')
				}
				weight_sums[output_feature] += i64(weight)
			}
		}
	} else {
		weight_sums = cached_weight_sums.clone()
		if weight_sums.len != out_features {
			return error('cached quantized Linear weight sums do not match output features')
		}
	}
	mut transposed_weights := []i8{}
	if preuploaded_b == none {
		transposed_weights = []i8{len: weights.size}
		for output_feature in 0 .. out_features {
			weight_start := output_feature * in_features
			for feature in 0 .. in_features {
				transposed_weights[feature * out_features + output_feature] = weight_values[weight_start + feature]
			}
		}
	}
	mut input_sums := []i64{len: rows}
	for row in 0 .. rows {
		for feature in 0 .. in_features {
			input_sums[row] += i64(input_values[row * in_features + feature])
		}
	}

	mut output_scales := []f64{len: out_features}
	for output_feature in 0 .. out_features {
		scale := input_params.scale * weight_params.scales[output_feature]
		if !math.is_finite(scale) || scale <= 0 {
			return error('combined quantization scale for output feature ${output_feature} is not finite and positive')
		}
		output_scales[output_feature] = scale
	}
	accumulators := if preuploaded_b == none {
		cuda_compute.gemm_i8_i64_cuda(dev, input_values, transposed_weights, rows, out_features,
			in_features)!
	} else {
		cuda_compute.gemm_i8_i64_cuda_with_b(dev, input_values, rows, out_features, in_features,
			preuploaded_b)!
	}
	mut output := []f32{len: rows * out_features}
	for row in 0 .. rows {
		for output_feature in 0 .. out_features {
			index := row * out_features + output_feature
			weight_zero_point := i64(weight_params.zero_points[output_feature])
			centered_accumulator := accumulators[index] -
				weight_zero_point * input_sums[row] -
				i64(input_params.zero_point) * weight_sums[output_feature] +
				i64(in_features) * i64(input_params.zero_point) * weight_zero_point
			output[index] = f32(f64(centered_accumulator) * output_scales[output_feature] +
				f64(bias_values[output_feature]))
		}
	}
	mut output_shape := input.shape[..input.shape.len - 1].clone()
	output_shape << out_features
	return vtl.from_array[f32](output, output_shape)!
}

// CudaQuantizedLinearLayer owns an immutable snapshot of INT8 weights on the CUDA device.
pub struct CudaQuantizedLinearLayer {
mut:
	layer          &QuantizedLinearLayer
	device         &cuda.CudaDevice
	device_weights &cuda_compute.CudaI8Matrix
	weight_sums    []i64
}

// upload_quantized_linear_layer_cuda snapshots and uploads weights for repeated inference.
pub fn upload_quantized_linear_layer_cuda(dev &cuda.CudaDevice, layer &QuantizedLinearLayer) !&CudaQuantizedLinearLayer {
	if layer.weights.rank() != 2 || layer.weights.shape[0] <= 0
		|| layer.weights.shape[1] <= 0 {
		return error('CUDA quantized Linear weights must have non-zero shape [out_features, in_features]')
	}
	out_features := layer.weights.shape[0]
	in_features := layer.weights.shape[1]
	if out_features > max_int / in_features {
		return error('CUDA quantized Linear weight shape exceeds supported integer sizes')
	}
	validate_params(layer.input_params)!
	weight_axis := normalize_axis(layer.weight_params.axis, layer.weights.rank())!
	if weight_axis != 0 {
		return error('quantized Linear weight parameters must use output-feature axis 0')
	}
	validate_channel_params(layer.weight_params, out_features)!
	weight_values := layer.weights.to_array()
	if weight_values.len != out_features * in_features {
		return error('quantized Linear weight storage does not match its shape')
	}
	bias_is_vector := layer.bias.rank() == 1 && layer.bias.shape[0] == out_features
	bias_is_row := layer.bias.rank() == 2 && layer.bias.shape[0] == 1
		&& layer.bias.shape[1] == out_features
	if !bias_is_vector && !bias_is_row {
		return error('quantized Linear bias must have shape [out_features] or [1, out_features]')
	}
	bias_values := layer.bias.to_array()
	if bias_values.len != out_features {
		return error('quantized Linear bias storage does not match its shape')
	}
	mut weight_sums := []i64{len: out_features}
	mut transposed_weights := []i8{len: layer.weights.size}
	for output_feature in 0 .. out_features {
		for feature in 0 .. in_features {
			index := output_feature * in_features + feature
			weight := weight_values[index]
			if weight < layer.weight_params.qmin || weight > layer.weight_params.qmax {
				return error('quantized weight at logical index ${index} is outside [${layer.weight_params.qmin}, ${layer.weight_params.qmax}]')
			}
			transposed_weights[feature * out_features + output_feature] = weight
			weight_sums[output_feature] += i64(weight)
		}
	}
	weight_params := PerChannelParams{
		scales:      layer.weight_params.scales.clone()
		zero_points: layer.weight_params.zero_points.clone()
		axis:        layer.weight_params.axis
		qmin:        layer.weight_params.qmin
		qmax:        layer.weight_params.qmax
	}
	weights_copy := vtl.from_array[i8](weight_values, layer.weights.shape.clone())!
	bias_copy := vtl.from_array[f32](bias_values, layer.bias.shape.clone())!
	layer_copy := &QuantizedLinearLayer{
		weights:       weights_copy
		weight_params: weight_params
		input_params:  layer.input_params
		bias:          bias_copy
	}
	device_weights := cuda_compute.upload_i8_matrix_cuda(dev, transposed_weights, in_features,
		out_features)!
	return &CudaQuantizedLinearLayer{
		layer:          layer_copy
		device:         dev
		device_weights: device_weights
		weight_sums:    weight_sums
	}
}

// forward runs the layer using its retained CUDA weight matrix.
pub fn (layer &CudaQuantizedLinearLayer) forward(input &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	return linear_int8_f32_cuda_impl(layer.device, input, layer.layer.weights,
		layer.layer.input_params, layer.layer.weight_params, layer.layer.bias,
		layer.device_weights, layer.weight_sums)
}

// release frees the retained device-side weight matrix.
pub fn (mut layer CudaQuantizedLinearLayer) release() ! {
	layer.device_weights.release()!
}
