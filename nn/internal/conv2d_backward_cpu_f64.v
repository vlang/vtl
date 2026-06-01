module internal

import vtl

// conv2d_backward_cpu_f64 computes Conv2D gradients on CPU (reference).
pub fn conv2d_backward_cpu_f64(grad_out &vtl.Tensor[f64], input &vtl.Tensor[f64], weight &vtl.Tensor[f64],
	bias &vtl.Tensor[f64], kernel_size []int, config Conv2DConfig) ![]&vtl.Tensor[f64] {
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

	out_h := grad_out.shape[2]
	out_w := grad_out.shape[3]

	mut d_input := vtl.zeros_like[f64](input)
	for b in 0 .. batch {
		for g in 0 .. groups {
			g_in_ch := in_ch / groups
			g_out_ch := out_ch / groups
			for oc in 0 .. g_out_ch {
				global_oc := g * g_out_ch + oc
				for oh in 0 .. out_h {
					for ow in 0 .. out_w {
						goh := grad_out.get([b, global_oc, oh, ow])
						for ic in 0 .. g_in_ch {
							for kh in 0 .. k_h {
								for kw in 0 .. k_w {
									ih := oh * stride_h - pad_h + kh * dil_h
									iw := ow * stride_w - pad_w + kw * dil_w
									if ih >= 0 && ih < in_h && iw >= 0 && iw < in_w {
										w_val := weight.get([global_oc, ic, kh, kw])
										d_input.set([b, g * g_in_ch + ic, ih, iw],

											d_input.get([b, g * g_in_ch + ic, ih, iw]) + goh * w_val)
									}
								}
							}
						}
					}
				}
			}
		}
	}

	mut d_weight := vtl.zeros_like[f64](weight)
	for oc in 0 .. out_ch {
		for ic in 0 .. in_ch {
			for kh in 0 .. k_h {
				for kw in 0 .. k_w {
					mut sum := 0.0
					for b in 0 .. batch {
						for oh in 0 .. out_h {
							for ow in 0 .. out_w {
								ih := oh * stride_h - pad_h + kh * dil_h
								iw := ow * stride_w - pad_w + kw * dil_w
								if ih >= 0 && ih < in_h && iw >= 0 && iw < in_w {
									sum += input.get([b, ic, ih, iw]) * grad_out.get([
										b,
										oc,
										oh,
										ow,
									])
								}
							}
						}
					}
					d_weight.set([oc, ic, kh, kw], sum)
				}
			}
		}
	}

	mut d_bias := vtl.zeros[f64]([1, out_ch])
	for oc in 0 .. out_ch {
		mut sum := 0.0
		for b in 0 .. batch {
			for oh in 0 .. out_h {
				for ow in 0 .. out_w {
					sum += grad_out.get([b, oc, oh, ow])
				}
			}
		}
		d_bias.set([0, oc], sum)
	}
	_ = bias
	return [d_input, d_weight, d_bias]
}
