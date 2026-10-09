module internal

import vtl

fn avgpool2d_output_shape[T](input &vtl.Tensor[T], kernel []int, padding []int, stride []int) ![]int {
	if input.rank() != 4 {
		return error('avgpool2d expects a 4D input tensor')
	}
	if kernel.len != 2 || padding.len != 2 || stride.len != 2 {
		return error('avgpool2d expects 2D kernel, padding, and stride')
	}
	if kernel[0] <= 0 || kernel[1] <= 0 || stride[0] <= 0 || stride[1] <= 0 || padding[0] < 0
		|| padding[1] < 0 {
		return error('avgpool2d expects positive kernel and stride and non-negative padding')
	}
	h := input.shape[2]
	w := input.shape[3]
	h_numerator := h + 2 * padding[0] - kernel[0]
	w_numerator := w + 2 * padding[1] - kernel[1]
	if h_numerator < 0 || w_numerator < 0 {
		return error('avgpool2d parameters produce an empty output')
	}
	out_h := h_numerator / stride[0] + 1
	out_w := w_numerator / stride[1] + 1
	if out_h <= 0 || out_w <= 0 {
		return error('avgpool2d parameters produce an empty output')
	}
	return [input.shape[0], input.shape[1], out_h, out_w]
}

// avgpool2d_forward exposes this operation as part of the public API.
pub fn avgpool2d_forward[T](input &vtl.Tensor[T], kernel []int, padding []int, stride []int) !&vtl.Tensor[T] {
	output_shape := avgpool2d_output_shape[T](input, kernel, padding, stride)!
	n := input.shape[0]
	c := input.shape[1]
	h := input.shape[2]
	w := input.shape[3]
	k_h := kernel[0]
	k_w := kernel[1]
	p_h := padding[0]
	p_w := padding[1]
	s_h := stride[0]
	s_w := stride[1]
	out_h := output_shape[2]
	out_w := output_shape[3]
	mut output := vtl.zeros[T](output_shape)
	for nn in 0 .. n {
		for cc in 0 .. c {
			for oh in 0 .. out_h {
				for ow in 0 .. out_w {
					mut sum := f64(0)
					for kh in 0 .. k_h {
						ih := oh * s_h - p_h + kh
						if ih < 0 || ih >= h {
							continue
						}
						for kw in 0 .. k_w {
							iw := ow * s_w - p_w + kw
							if iw < 0 || iw >= w {
								continue
							}
							sum += f64(input.get([nn, cc, ih, iw]))
						}
					}
					output.set([nn, cc, oh, ow], vtl.cast[T](sum / f64(k_h * k_w)))
				}
			}
		}
	}
	return output
}

// avgpool2d_backward exposes this operation as part of the public API.
pub fn avgpool2d_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T], kernel []int, padding []int, stride []int) !&vtl.Tensor[T] {
	output_shape := avgpool2d_output_shape[T](input, kernel, padding, stride)!
	if grad_out.rank() != 4 {
		return error('avgpool2d backward expects a 4D gradient tensor')
	}
	n := input.shape[0]
	c := input.shape[1]
	h := input.shape[2]
	w := input.shape[3]
	if grad_out.shape != output_shape {
		return error('avgpool2d backward gradient shape does not match input and pooling parameters')
	}
	mut d_input := vtl.zeros_like[T](input)
	area := f64(kernel[0]) * f64(kernel[1])
	for batch in 0 .. n {
		for channel in 0 .. c {
			for out_h in 0 .. output_shape[2] {
				for out_w in 0 .. output_shape[3] {
					contribution := f64(grad_out.get([batch, channel, out_h, out_w])) / area
					for kernel_h in 0 .. kernel[0] {
						in_h := out_h * stride[0] - padding[0] + kernel_h
						if in_h < 0 || in_h >= h {
							continue
						}
						for kernel_w in 0 .. kernel[1] {
							in_w := out_w * stride[1] - padding[1] + kernel_w
							if in_w < 0 || in_w >= w {
								continue
							}
							index := [batch, channel, in_h, in_w]
							value := f64(d_input.get(index)) + contribution
							d_input.set(index, vtl.cast[T](value))
						}
					}
				}
			}
		}
	}
	return d_input
}

// global_avgpool2d_forward exposes this operation as part of the public API.
pub fn global_avgpool2d_forward[T](input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	n := input.shape[0]
	c := input.shape[1]
	h := input.shape[2]
	w := input.shape[3]
	mut output := vtl.zeros[T]([n, c, 1, 1])
	for nn in 0 .. n {
		for cc in 0 .. c {
			mut sum := f64(0)
			for hh in 0 .. h {
				for ww in 0 .. w {
					sum += f64(input.get([nn, cc, hh, ww]))
				}
			}
			output.set([nn, cc, 0, 0], vtl.cast[T](sum / f64(h * w)))
		}
	}
	return output
}

// global_avgpool2d_backward exposes this operation as part of the public API.
pub fn global_avgpool2d_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	n := input.shape[0]
	c := input.shape[1]
	h := input.shape[2]
	w := input.shape[3]
	mut d_input := vtl.zeros_like[T](input)
	for nn in 0 .. n {
		for cc in 0 .. c {
			grad_val := f64(grad_out.get([nn, cc, 0, 0]))
			for hh in 0 .. h {
				for ww in 0 .. w {
					d_input.set([nn, cc, hh, ww], vtl.cast[T](grad_val / f64(h * w)))
				}
			}
		}
	}
	return d_input
}
