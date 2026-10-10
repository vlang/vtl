module internal

import vtl

// conv2d_forward_cpu_f32 uses direct indexing for compact row-major tensors
// and preserves the stride-aware reference path for views.
pub fn conv2d_forward_cpu_f32(input &vtl.Tensor[f32],
	weight &vtl.Tensor[f32],
	bias &vtl.Tensor[f32],
	kernel_size []int,
	config Conv2DConfig) !&vtl.Tensor[f32] {
	validate_conv2d(input, weight, bias, kernel_size, config)!
	if input.is_row_major_contiguous() && input.data.data.len == input.size
		&& weight.is_row_major_contiguous() && weight.data.data.len == weight.size
		&& bias.is_row_major_contiguous() && bias.data.data.len == bias.size {
		return conv2d_forward_cpu_f32_contiguous(input, weight, bias, kernel_size, config)
	}
	return conv2d_forward_cpu_f32_reference(input, weight, bias, kernel_size, config)
}

// conv2d_forward_cpu_f32_contiguous avoids allocating index arrays for each
// input, weight, and output access in compact NCHW/OIHW tensors.
@[direct_array_access]
fn conv2d_forward_cpu_f32_contiguous(input &vtl.Tensor[f32], weight &vtl.Tensor[f32],
	bias &vtl.Tensor[f32], kernel_size []int, config Conv2DConfig) !&vtl.Tensor[f32] {
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
	mut output := vtl.zeros[f32]([batch, out_ch, out_h, out_w])
	for b in 0 .. batch {
		for g in 0 .. groups {
			for oc in 0 .. g_out_ch {
				global_oc := g * g_out_ch + oc
				for oh in 0 .. out_h {
					for ow in 0 .. out_w {
						mut sum := f32(0)
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

// conv2d_forward_cpu_f32_reference is the stride-aware implementation.
fn conv2d_forward_cpu_f32_reference(input &vtl.Tensor[f32], weight &vtl.Tensor[f32],
	bias &vtl.Tensor[f32], kernel_size []int, config Conv2DConfig) !&vtl.Tensor[f32] {
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

	mut output := vtl.zeros[f32]([batch, out_ch, out_h, out_w])

	for b in 0 .. batch {
		for g in 0 .. groups {
			g_in_ch := in_ch / groups
			g_out_ch := out_ch / groups
			for oc in 0 .. g_out_ch {
				global_oc := g * g_out_ch + oc
				for oh in 0 .. out_h {
					for ow in 0 .. out_w {
						mut sum := f32(0)
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
