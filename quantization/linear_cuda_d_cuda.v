module quantization

import math
import vtl
import vsl.cuda
import vsl.cuda.compute as cuda_compute

// linear_int8_f32_cuda runs affine-INT8 Linear inference with cuBLAS.
// The result is a CPU-resident f32 tensor; the caller owns the CUDA device.
pub fn linear_int8_f32_cuda(dev &cuda.CudaDevice, input &vtl.Tensor[f32], weights &vtl.Tensor[i8], input_params QuantizationParams, weight_params PerChannelParams, bias &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
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
	weight_values := weights.to_array()
	bias_values := bias.to_array()
	if input_values.len != input.size || weight_values.len != weights.size
		|| bias_values.len != out_features {
		return error('quantized linear tensor storage does not match its shape')
	}

	mut transposed_weights := []i8{len: weights.size}
	mut weight_sums := []i64{len: out_features}
	for output_feature in 0 .. out_features {
		weight_start := output_feature * in_features
		for feature in 0 .. in_features {
			weight := weight_values[weight_start + feature]
			if weight < weight_params.qmin || weight > weight_params.qmax {
				return error('quantized weight at logical index ${weight_start + feature} is outside [${weight_params.qmin}, ${weight_params.qmax}]')
			}
			transposed_weights[feature * out_features + output_feature] = weight
			weight_sums[output_feature] += i64(weight)
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
	accumulators := cuda_compute.gemm_i8_i64_cuda(dev, input_values, transposed_weights, rows,
		out_features, in_features)!
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
