module internal

import vtl

// conv2d_forward_cpu_f64 is the reference CPU implementation (nested loops).
pub fn conv2d_forward_cpu_f64(input &vtl.Tensor[f64],
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
