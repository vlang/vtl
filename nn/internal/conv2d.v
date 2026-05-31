module internal

import vtl

// Conv2DConfig mirrors vtl.nn.layers.Conv2DConfig to avoid import cycle.
pub struct Conv2DConfig {
pub:
	padding  []int = [0, 0]
	stride   []int = [1, 1]
	dilation []int = [1, 1]
	groups   int   = 1
}

// conv2d_forward implements the forward pass of 2D convolution.
// input: [batch, in_ch, H, W]
// weight: [out_ch, in_ch/groups, k_h, k_w]
// bias: [1, out_ch]
// config: padding, stride, dilation, groups
// returns: [batch, out_ch, out_H, out_W]
pub fn conv2d_forward[T](input &vtl.Tensor[T],
	weight &vtl.Tensor[T],
	bias &vtl.Tensor[T],
	kernel_size []int,
	config Conv2DConfig) !&vtl.Tensor[T] {
	if sizeof(T) == 8 {
		return conv2d_forward_f64(unsafe { &vtl.Tensor[f64](input) },
			unsafe { &vtl.Tensor[f64](weight) }, unsafe { &vtl.Tensor[f64](bias) }, kernel_size,
			config)
	}
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

	mut output := vtl.zeros[T]([batch, out_ch, out_h, out_w])

	for b in 0 .. batch {
		for g in 0 .. groups {
			g_in_ch := in_ch / groups
			g_out_ch := out_ch / groups
			for oc in 0 .. g_out_ch {
				global_oc := g * g_out_ch + oc
				for oh in 0 .. out_h {
					for ow in 0 .. out_w {
						// Compute correlation at this output position
						mut sum := f64(0)
						for ic in 0 .. g_in_ch {
							for kh in 0 .. k_h {
								for kw in 0 .. k_w {
									ih := oh * stride_h - pad_h + kh * dil_h
									iw := ow * stride_w - pad_w + kw * dil_w
									if ih >= 0 && ih < in_h && iw >= 0 && iw < in_w {
										in_val := f64(input.get([b, g * g_in_ch + ic, ih, iw]))
										w_val := f64(weight.get([global_oc, ic, kh, kw]))
										sum += in_val * w_val
									}
								}
							}
						}
						out_val := sum + f64(bias.get([0, global_oc]))
						output.set([b, global_oc, oh, ow], vtl.cast[T](out_val))
					}
				}
			}
		}
	}
	return output
}

// conv2d_backward computes gradients for input, weight, bias.
// Returns [d_input, d_weight, d_bias].
pub fn conv2d_backward[T](grad_out &vtl.Tensor[T],
	input &vtl.Tensor[T],
	weight &vtl.Tensor[T],
	bias &vtl.Tensor[T],
	kernel_size []int,
	config Conv2DConfig) ![]&vtl.Tensor[T] {
	if sizeof(T) == 8 {
		return conv2d_backward_f64(unsafe { &vtl.Tensor[f64](grad_out) },
			unsafe { &vtl.Tensor[f64](input) }, unsafe { &vtl.Tensor[f64](weight) },
			unsafe { &vtl.Tensor[f64](bias) }, kernel_size, config)
	}
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

	// d_input: accumulate gradients from all output positions that touched each input pixel
	mut d_input := vtl.zeros_like[T](input)
	for b in 0 .. batch {
		for g in 0 .. groups {
			g_in_ch := in_ch / groups
			g_out_ch := out_ch / groups
			for oc in 0 .. g_out_ch {
				global_oc := g * g_out_ch + oc
				for oh in 0 .. out_h {
					for ow in 0 .. out_w {
						goh := f64(grad_out.get([b, global_oc, oh, ow]))
						for ic in 0 .. g_in_ch {
							for kh in 0 .. k_h {
								for kw in 0 .. k_w {
									ih := oh * stride_h - pad_h + kh * dil_h
									iw := ow * stride_w - pad_w + kw * dil_w
									if ih >= 0 && ih < in_h && iw >= 0 && iw < in_w {
										w_val := f64(weight.get([global_oc, ic, kh, kw]))
										d_input.set([b, g * g_in_ch + ic, ih, iw],
											d_input.get([b, g * g_in_ch + ic, ih, iw]) +
											vtl.cast[T](goh * w_val))
									}
								}
							}
						}
					}
				}
			}
		}
	}

	// d_weight: correlate input with gradient output
	mut d_weight := vtl.zeros_like[T](weight)
	for oc in 0 .. out_ch {
		for ic in 0 .. in_ch {
			for kh in 0 .. k_h {
				for kw in 0 .. k_w {
					mut sum := f64(0)
					for b in 0 .. batch {
						for oh in 0 .. out_h {
							for ow in 0 .. out_w {
								ih := oh * stride_h - pad_h + kh * dil_h
								iw := ow * stride_w - pad_w + kw * dil_w
								if ih >= 0 && ih < in_h && iw >= 0 && iw < in_w {
									sum += f64(input.get([b, ic, ih, iw])) * f64(grad_out.get([
										b,
										oc,
										oh,
										ow,
									]))
								}
							}
						}
					}
					d_weight.set([oc, ic, kh, kw], vtl.cast[T](sum))
				}
			}
		}
	}

	// d_bias: sum gradients over batch and spatial dimensions
	mut d_bias := vtl.zeros[T]([1, out_ch])
	for oc in 0 .. out_ch {
		mut sum := f64(0)
		for b in 0 .. batch {
			for oh in 0 .. out_h {
				for ow in 0 .. out_w {
					sum += f64(grad_out.get([b, oc, oh, ow]))
				}
			}
		}
		d_bias.set([0, oc], vtl.cast[T](sum))
	}

	return [d_input, d_weight, d_bias]
}
