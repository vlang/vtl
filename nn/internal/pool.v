module internal

import vtl

pub fn avgpool2d_forward[T](input &vtl.Tensor[T], kernel []int, padding []int, stride []int) !&vtl.Tensor[T] {
	n := input.shape[0]
	c := input.shape[1]
	h := input.shape[2]
	w := input.shape[3]
	kH := kernel[0]
	kW := kernel[1]
	pH := padding[0]
	pW := padding[1]
	sH := stride[0]
	sW := stride[1]
	out_h := (h + 2 * pH - kH) / sH + 1
	out_w := (w + 2 * pW - kW) / sW + 1
	mut output := vtl.zeros[T]([n, c, out_h, out_w])
	for nn in 0 .. n {
		for cc in 0 .. c {
			for oh in 0 .. out_h {
				for ow in 0 .. out_w {
					mut sum := f64(0)
					for kh in 0 .. kH {
						ih := oh * sH - pH + kh
						if ih < 0 || ih >= h {
							continue
						}
						for kw in 0 .. kW {
							iw := ow * sW - pW + kw
							if iw < 0 || iw >= w {
								continue
							}
							sum += f64(input.get([nn, cc, ih, iw]))
						}
					}
					output.set([nn, cc, oh, ow], vtl.cast[T](sum / f64(kH * kW)))
				}
			}
		}
	}
	return output
}

pub fn avgpool2d_backward[T](grad_out &vtl.Tensor[T], kernel []int, padding []int, stride []int) !&vtl.Tensor[T] {
	n := grad_out.shape[0]
	c := grad_out.shape[1]
	h := grad_out.shape[2]
	w := grad_out.shape[3]
	_ = kernel
	_ = padding
	_ = stride
	mut d_input := vtl.zeros[T]([n, c, h, w])
	return d_input
}

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
