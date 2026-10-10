module internal

import vtl

// conv2d_forward_cpu_f64 runs the contiguous fast path when tensors use
// row-major storage, and preserves the reference implementation for views.
pub fn conv2d_forward_cpu_f64(input &vtl.Tensor[f64],
	weight &vtl.Tensor[f64],
	bias &vtl.Tensor[f64],
	kernel_size []int,
	config Conv2DConfig) !&vtl.Tensor[f64] {
	validate_conv2d(input, weight, bias, kernel_size, config)!
	if input.is_row_major_contiguous() && input.data.data.len == input.size
		&& weight.is_row_major_contiguous() && weight.data.data.len == weight.size
		&& bias.is_row_major_contiguous() && bias.data.data.len == bias.size {
		return conv2d_forward_cpu_f64_contiguous(input, weight, bias, kernel_size, config)
	}
	return conv2d_forward_cpu_f64_reference(input, weight, bias, kernel_size, config)
}

fn validate_conv2d_tensor[T](tensor &vtl.Tensor[T], name string) ! {
	if tensor.shape.len != tensor.strides.len {
		return error('${name} shape and strides must have the same rank')
	}
	mut expected_size := 1
	mut min_offset := 0
	mut max_offset := 0
	for i, dim in tensor.shape {
		if dim < 0 {
			return error('${name} dimensions must not be negative')
		}
		stride := tensor.strides[i]
		if stride < 0 {
			if stride == min_int
				|| (dim > 0 && dim - 1 > max_int / -stride) {
				return error('${name} strides are invalid or too large')
			}
			span := if dim > 0 { (dim - 1) * -stride } else { 0 }
			adjustment := if dim > 0 { dim - 1 } else { 0 }
			if span > max_int + min_offset || adjustment > max_int - max_offset {
				return error('${name} storage span is too large')
			}
			min_offset -= span
			max_offset += adjustment
		} else {
			if stride != 0 && dim > 0 && dim - 1 > max_int / stride {
				return error('${name} strides are invalid or too large')
			}
			span := if dim > 0 { (dim - 1) * stride } else { 0 }
			if span > max_int - max_offset {
				return error('${name} storage span is too large')
			}
			max_offset += span
		}
		if dim != 0 && expected_size > max_int / dim {
			return error('${name} shape is too large')
		}
		expected_size *= dim
	}
	if tensor.size != expected_size {
		return error('${name} size does not match its shape')
	}
	min_offset = if min_offset < 0 { tensor.size - 1 + min_offset } else { min_offset }
	max_offset = if max_offset < 0 { tensor.size - 1 + max_offset } else { max_offset }
	if isnil(tensor.data) || tensor.data.data.len < tensor.size
		|| (tensor.size > 0 && (min_offset < 0 || max_offset >= tensor.data.data.len)) {
		return error('${name} storage is smaller than its declared size')
	}
}

fn validate_conv2d[T](input &vtl.Tensor[T], weight &vtl.Tensor[T], bias &vtl.Tensor[T],
	kernel_size []int, config Conv2DConfig) ! {
	validate_conv2d_tensor[T](input, 'input')!
	validate_conv2d_tensor[T](weight, 'weight')!
	validate_conv2d_tensor[T](bias, 'bias')!
	if input.shape.len != 4 || weight.shape.len != 4 || bias.shape.len != 2 {
		return error('Conv2D expects input and weight rank 4 and bias rank 2')
	}
	if kernel_size.len != 2 || config.padding.len != 2 || config.stride.len != 2
		|| config.dilation.len != 2 {
		return error('Conv2D kernel_size, padding, stride, and dilation must have two values')
	}
	if kernel_size[0] <= 0 || kernel_size[1] <= 0 || config.stride[0] <= 0
		|| config.stride[1] <= 0 || config.dilation[0] <= 0 || config.dilation[1] <= 0
		|| config.padding[0] < 0 || config.padding[1] < 0 || config.groups <= 0 {
		return error('Conv2D kernel, stride, dilation, and groups must be positive; padding must be non-negative')
	}
	in_ch := input.shape[1]
	out_ch := weight.shape[0]
	if in_ch == 0 || out_ch == 0 || in_ch % config.groups != 0 || out_ch % config.groups != 0 {
		return error('Conv2D groups must divide non-zero input and output channels')
	}
	if weight.shape[1] != in_ch / config.groups || weight.shape[2] != kernel_size[0]
		|| weight.shape[3] != kernel_size[1] || bias.shape[0] != 1 || bias.shape[1] != out_ch {
		return error('Conv2D weight or bias shape does not match input, kernel, and groups')
	}
	if input.shape[0] <= 0 || input.shape[2] <= 0 || input.shape[3] <= 0 {
		return error('Conv2D batch and input dimensions must be positive')
	}
	if config.padding[0] > (max_int - input.shape[2]) / 2
		|| config.padding[1] > (max_int - input.shape[3]) / 2
		|| kernel_size[0] - 1 > (max_int - 1) / config.dilation[0]
		|| kernel_size[1] - 1 > (max_int - 1) / config.dilation[1] {
		return error('Conv2D dimensions exceed supported integer sizes')
	}
	effective_h := config.dilation[0] * (kernel_size[0] - 1) + 1
	effective_w := config.dilation[1] * (kernel_size[1] - 1) + 1
	if input.shape[2] + 2 * config.padding[0] < effective_h
		|| input.shape[3] + 2 * config.padding[1] < effective_w {
		return error('Conv2D effective kernel must fit the padded input')
	}
	out_h := (input.shape[2] + 2 * config.padding[0] - effective_h) / config.stride[0] + 1
	out_w := (input.shape[3] + 2 * config.padding[1] - effective_w) / config.stride[1] + 1
	mut output_size := input.shape[0]
	for dim in [out_ch, out_h, out_w] {
		if dim != 0 && output_size > max_int / dim {
			return error('Conv2D output shape is too large')
		}
		output_size *= dim
	}
}

// conv2d_forward_cpu_f64_contiguous applies NCHW/OIHW convolution to compact
// row-major tensors without allocating index arrays for each tensor access.
@[direct_array_access]
fn conv2d_forward_cpu_f64_contiguous(input &vtl.Tensor[f64], weight &vtl.Tensor[f64],
	bias &vtl.Tensor[f64], kernel_size []int, config Conv2DConfig) !&vtl.Tensor[f64] {
	batch := input.shape[0]
	in_ch := input.shape[1]
	in_h := input.shape[2]
	in_w := input.shape[3]
	out_ch := weight.shape[0]
	k_h := kernel_size[0]
	k_w := kernel_size[1]
	pad_h := config.padding[0]
	pad_w := config.padding[1]
	stride_h := config.stride[0]
	stride_w := config.stride[1]
	dil_h := config.dilation[0]
	dil_w := config.dilation[1]
	groups := config.groups

	out_h := (in_h + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1
	out_w := (in_w + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1
	g_in_ch := in_ch / groups
	g_out_ch := out_ch / groups
	input_data := input.data.data
	weight_data := weight.data.data
	bias_data := bias.data.data
	mut output := vtl.zeros[f64]([batch, out_ch, out_h, out_w])
	for b in 0 .. batch {
		for g in 0 .. groups {
			for oc in 0 .. g_out_ch {
				global_oc := g * g_out_ch + oc
				for oh in 0 .. out_h {
					for ow in 0 .. out_w {
						mut sum := 0.0
						for ic in 0 .. g_in_ch {
							input_channel := g * g_in_ch + ic
							for kh in 0 .. k_h {
								ih := oh * stride_h - pad_h + kh * dil_h
								if ih < 0 || ih >= in_h {
									continue
								}
								for kw in 0 .. k_w {
									iw := ow * stride_w - pad_w + kw * dil_w
									if iw >= 0 && iw < in_w {
										input_index := ((b * in_ch + input_channel) * in_h + ih) * in_w + iw
										weight_index := ((global_oc * g_in_ch + ic) * k_h + kh) * k_w + kw
										sum += input_data[input_index] * weight_data[weight_index]
									}
								}
							}
						}
						output_index := ((b * out_ch + global_oc) * out_h + oh) * out_w + ow
						output.data.data[output_index] = sum + bias_data[global_oc]
					}
				}
			}
		}
	}
	return output
}

// conv2d_forward_cpu_f64_reference retains the general strided-tensor fallback.
fn conv2d_forward_cpu_f64_reference(input &vtl.Tensor[f64],
	weight &vtl.Tensor[f64],
	bias &vtl.Tensor[f64],
	kernel_size []int,
	config Conv2DConfig) !&vtl.Tensor[f64] {
	batch := input.shape[0]
	in_ch := input.shape[1]
	in_h := input.shape[2]
	in_w := input.shape[3]
	out_ch := weight.shape[0]
	k_h := kernel_size[0]
	k_w := kernel_size[1]
	pad_h := config.padding[0]
	pad_w := config.padding[1]
	stride_h := config.stride[0]
	stride_w := config.stride[1]
	dil_h := config.dilation[0]
	dil_w := config.dilation[1]
	groups := config.groups

	out_h := (in_h + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1
	out_w := (in_w + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1

	mut output := vtl.zeros[f64]([batch, out_ch, out_h, out_w])

	for b in 0 .. batch {
		for g in 0 .. groups {
			g_in_ch := in_ch / groups
			g_out_ch := out_ch / groups
			for oc in 0 .. g_out_ch {
				global_oc := g * g_out_ch + oc
				for oh in 0 .. out_h {
					for ow in 0 .. out_w {
						mut sum := 0.0
						for ic in 0 .. g_in_ch {
							for kh in 0 .. k_h {
								for kw in 0 .. k_w {
									ih := oh * stride_h - pad_h + kh * dil_h
									iw := ow * stride_w - pad_w + kw * dil_w
									if ih >= 0 && ih < in_h && iw >= 0 && iw < in_w {
										sum += input.get([b, g * g_in_ch + ic, ih, iw]) * weight.get([
											global_oc,
											ic,
											kh,
											kw,
										])
									}
								}
							}
						}
						output.set([b, global_oc, oh, ow], sum + bias.get([0, global_oc]))
					}
				}
			}
		}
	}
	return output
}
