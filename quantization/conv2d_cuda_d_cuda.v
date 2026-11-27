module quantization

import math
import vtl
import vsl.cuda
import vsl.cuda.compute as cuda_compute

const max_im2col_workspace_bytes = 64 * 1024 * 1024

// conv2d_int8_f32_cuda runs affine-INT8 Conv2D inference with cuBLAS.
// The result is a CPU-resident f32 tensor; the caller owns the CUDA device.
pub fn conv2d_int8_f32_cuda(dev &cuda.CudaDevice, input &vtl.Tensor[f32], weights &vtl.Tensor[i8], input_params QuantizationParams, weight_params PerChannelParams, bias &vtl.Tensor[f32], kernel_size []int, config QuantizedConv2DConfig) !&vtl.Tensor[f32] {
	if input.rank() != 4 || weights.rank() != 4 || bias.rank() != 2 {
		return error('quantized CUDA Conv2D expects input and weights rank 4 and bias rank 2')
	}
	if kernel_size.len != 2 || config.padding.len != 2 || config.stride.len != 2
		|| config.dilation.len != 2 {
		return error('quantized CUDA Conv2D geometry must have two values per dimension')
	}
	if kernel_size[0] <= 0 || kernel_size[1] <= 0 || config.stride[0] <= 0
		|| config.stride[1] <= 0 || config.dilation[0] <= 0 || config.dilation[1] <= 0
		|| config.padding[0] < 0 || config.padding[1] < 0 || config.groups <= 0 {
		return error('quantized CUDA Conv2D kernel, stride, dilation, and groups must be positive; padding must be non-negative')
	}
	batch := input.shape[0]
	in_channels := input.shape[1]
	in_h := input.shape[2]
	in_w := input.shape[3]
	out_channels := weights.shape[0]
	if batch <= 0 || in_channels <= 0 || out_channels <= 0 || in_h <= 0 || in_w <= 0
		|| in_channels % config.groups != 0 || out_channels % config.groups != 0 {
		return error('quantized CUDA Conv2D groups must divide positive batch and channel dimensions')
	}
	in_channels_per_group := in_channels / config.groups
	out_channels_per_group := out_channels / config.groups
	if weights.shape[1] != in_channels_per_group || weights.shape[2] != kernel_size[0]
		|| weights.shape[3] != kernel_size[1] || bias.shape[0] != 1
		|| bias.shape[1] != out_channels {
		return error('quantized CUDA Conv2D weight or bias shape does not match input, kernel, and groups')
	}
	if config.padding[0] > (max_int - in_h) / 2
		|| config.padding[1] > (max_int - in_w) / 2
		|| kernel_size[0] - 1 > (max_int - 1) / config.dilation[0]
		|| kernel_size[1] - 1 > (max_int - 1) / config.dilation[1] {
		return error('quantized CUDA Conv2D dimensions exceed supported integer sizes')
	}
	effective_h := config.dilation[0] * (kernel_size[0] - 1) + 1
	effective_w := config.dilation[1] * (kernel_size[1] - 1) + 1
	padded_h := in_h + 2 * config.padding[0]
	padded_w := in_w + 2 * config.padding[1]
	if padded_h < effective_h || padded_w < effective_w {
		return error('quantized CUDA Conv2D effective kernel must fit the padded input')
	}
	out_h := (padded_h - effective_h) / config.stride[0] + 1
	out_w := (padded_w - effective_w) / config.stride[1] + 1
	if out_h > max_int / out_w || in_channels_per_group > max_int / kernel_size[0]
		|| in_channels_per_group * kernel_size[0] > max_int / kernel_size[1] {
		return error('quantized CUDA Conv2D dimensions exceed supported integer sizes')
	}
	positions := out_h * out_w
	features := in_channels_per_group * kernel_size[0] * kernel_size[1]
	mut output_size := batch
	for dimension in [out_channels, out_h, out_w] {
		if output_size > max_int / dimension {
			return error('quantized CUDA Conv2D output shape is too large')
		}
		output_size *= dimension
	}
	weight_axis := normalize_axis(weight_params.axis, weights.rank())!
	if weight_axis != 0 {
		return error('quantized CUDA Conv2D weight parameters must use output-channel axis 0')
	}
	validate_channel_params(weight_params, out_channels)!
	validate_params(input_params)!
	quantized_input := quantize_per_tensor_f32(input, input_params)!
	input_values := quantized_input.to_array()
	weight_values := weights.to_array()
	bias_values := bias.to_array()
	if input_values.len != input.size || weight_values.len != weights.size
		|| bias_values.len != out_channels {
		return error('quantized CUDA Conv2D tensor storage does not match its shape')
	}
	mut weight_sums := []i64{len: out_channels}
	for output_channel in 0 .. out_channels {
		weight_start := output_channel * features
		for feature in 0 .. features {
			weight := weight_values[weight_start + feature]
			if weight < weight_params.qmin || weight > weight_params.qmax {
				return error('quantized weight at logical index ${weight_start + feature} is outside [${weight_params.qmin}, ${weight_params.qmax}]')
			}
			weight_sums[output_channel] += i64(weight)
		}
	}
	mut output_scales := []f64{len: out_channels}
	for output_channel in 0 .. out_channels {
		scale := input_params.scale * weight_params.scales[output_channel]
		if !math.is_finite(scale) || scale <= 0 {
			return error('combined quantization scale for output channel ${output_channel} is not finite and positive')
		}
		output_scales[output_channel] = scale
	}

	mut output := []f32{len: output_size}
	positions_per_block := math.min(positions, math.max(1, max_im2col_workspace_bytes / features))
	for group in 0 .. config.groups {
		mut weight_matrix := []i8{len: features * out_channels_per_group}
		for output_channel_in_group in 0 .. out_channels_per_group {
			output_channel := group * out_channels_per_group + output_channel_in_group
			weight_start := output_channel * features
			for feature in 0 .. features {
				weight_matrix[feature * out_channels_per_group + output_channel_in_group] = weight_values[weight_start + feature]
			}
		}
		for batch_index in 0 .. batch {
			for position_start := 0; position_start < positions; position_start += positions_per_block {
				block_rows := math.min(positions_per_block, positions - position_start)
				mut input_matrix := []i8{len: block_rows * features}
				mut input_sums := []i64{len: block_rows}
				for feature in 0 .. features {
					input_channel_in_group := feature / (kernel_size[0] * kernel_size[1])
					kernel_offset := feature % (kernel_size[0] * kernel_size[1])
					kh := kernel_offset / kernel_size[1]
					kw := kernel_offset % kernel_size[1]
					input_channel := group * in_channels_per_group + input_channel_in_group
					for block_row in 0 .. block_rows {
						output_position := position_start + block_row
						oh := output_position / out_w
						ow := output_position % out_w
						ih := oh * config.stride[0] - config.padding[0] + kh *
							config.dilation[0]
						iw := ow * config.stride[1] - config.padding[1] + kw *
							config.dilation[1]
						value := if ih >= 0 && ih < in_h && iw >= 0 && iw < in_w {
							input_index := ((batch_index * in_channels + input_channel) *
								in_h + ih) * in_w + iw
							input_values[input_index]
						} else {
							i8(input_params.zero_point)
						}
						input_index := block_row * features + feature
						input_matrix[input_index] = value
						input_sums[block_row] += i64(value)
					}
				}
				accumulators := cuda_compute.gemm_i8_i64_cuda(dev, input_matrix,
					weight_matrix, block_rows, out_channels_per_group, features)!
				for block_row in 0 .. block_rows {
					output_position := position_start + block_row
					oh := output_position / out_w
					ow := output_position % out_w
					for output_channel_in_group in 0 .. out_channels_per_group {
						output_channel := group * out_channels_per_group +
							output_channel_in_group
						accumulator_index := block_row * out_channels_per_group +
							output_channel_in_group
						weight_zero_point := i64(weight_params.zero_points[output_channel])
						centered_accumulator := accumulators[accumulator_index] -
							weight_zero_point * input_sums[block_row] -
							i64(input_params.zero_point) * weight_sums[output_channel] +
							i64(features) * i64(input_params.zero_point) * weight_zero_point
						output_batch := batch_index * out_channels + output_channel
						output_index := (output_batch * out_h + oh) * out_w + ow
						output[output_index] = f32(f64(centered_accumulator) *
							output_scales[output_channel] + f64(bias_values[output_channel]))
					}
				}
			}
		}
	}
	return vtl.from_array[f32](output, [batch, out_channels, out_h, out_w])!
}
