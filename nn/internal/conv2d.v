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
		out := conv2d_forward_f64(unsafe { &vtl.Tensor[f64](input) },
			unsafe { &vtl.Tensor[f64](weight) }, unsafe { &vtl.Tensor[f64](bias) }, kernel_size,
			config)!
		return unsafe { &vtl.Tensor[T](out) }
	}
	out_f32 := conv2d_forward_f32(unsafe { &vtl.Tensor[f32](input) },
		unsafe { &vtl.Tensor[f32](weight) }, unsafe { &vtl.Tensor[f32](bias) }, kernel_size,
		config)!
	return unsafe { &vtl.Tensor[T](out_f32) }
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
		tensors := conv2d_backward_f64(unsafe { &vtl.Tensor[f64](grad_out) },
			unsafe { &vtl.Tensor[f64](input) }, unsafe { &vtl.Tensor[f64](weight) },
			unsafe { &vtl.Tensor[f64](bias) }, kernel_size, config)!
		mut out := []&vtl.Tensor[T]{cap: tensors.len}
		for t in tensors {
			out << unsafe { &vtl.Tensor[T](t) }
		}
		return out
	}
	tensors_f32 := conv2d_backward_f32(unsafe { &vtl.Tensor[f32](grad_out) },
		unsafe { &vtl.Tensor[f32](input) }, unsafe { &vtl.Tensor[f32](weight) },
		unsafe { &vtl.Tensor[f32](bias) }, kernel_size, config)!
	mut out_f32 := []&vtl.Tensor[T]{cap: tensors_f32.len}
	for t in tensors_f32 {
		out_f32 << unsafe { &vtl.Tensor[T](t) }
	}
	return out_f32
}
