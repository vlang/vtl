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

fn adaptive_avgpool2d_output_shape[T](input &vtl.Tensor[T], output_size []int) ![]int {
	if input.rank() != 4 {
		return error('adaptive_avgpool2d expects a 4D input tensor')
	}
	if output_size.len != 2 || output_size[0] <= 0 || output_size[1] <= 0 {
		return error('adaptive_avgpool2d expects two positive output dimensions')
	}
	if input.shape[2] <= 0 || input.shape[3] <= 0 {
		return error('adaptive_avgpool2d expects positive spatial dimensions')
	}
	return [input.shape[0], input.shape[1], output_size[0], output_size[1]]
}

// adaptive_avgpool2d_forward computes non-overlapping or overlapping adaptive
// mean regions whose boundaries are derived from the requested output shape.
pub fn adaptive_avgpool2d_forward[T](input &vtl.Tensor[T], output_size []int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool2d_output_shape[T](input, output_size)!
	mut output := vtl.zeros[T](output_shape)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_h in 0 .. output_size[0] {
				start_h := (out_h * input.shape[2]) / output_size[0]
				end_h := ((out_h + 1) * input.shape[2] + output_size[0] - 1) / output_size[0]
				for out_w in 0 .. output_size[1] {
					start_w := (out_w * input.shape[3]) / output_size[1]
					end_w := ((out_w + 1) * input.shape[3] + output_size[1] - 1) / output_size[1]
					mut sum := f64(0)
					for in_h in start_h .. end_h {
						for in_w in start_w .. end_w {
							sum += f64(input.get([batch, channel, in_h, in_w]))
						}
					}
					area := (end_h - start_h) * (end_w - start_w)
					output.set([batch, channel, out_h, out_w], vtl.cast[T](sum / f64(area)))
				}
			}
		}
	}
	return output
}

// adaptive_avgpool2d_backward distributes each output gradient over its
// corresponding adaptive mean region.
pub fn adaptive_avgpool2d_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T], output_size []int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool2d_output_shape[T](input, output_size)!
	if grad_out.shape != output_shape {
		return error('adaptive_avgpool2d backward gradient shape does not match input and output size')
	}
	mut d_input := vtl.zeros_like[T](input)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_h in 0 .. output_size[0] {
				start_h := (out_h * input.shape[2]) / output_size[0]
				end_h := ((out_h + 1) * input.shape[2] + output_size[0] - 1) / output_size[0]
				for out_w in 0 .. output_size[1] {
					start_w := (out_w * input.shape[3]) / output_size[1]
					end_w := ((out_w + 1) * input.shape[3] + output_size[1] - 1) / output_size[1]
					area := f64((end_h - start_h) * (end_w - start_w))
					contribution := f64(grad_out.get([batch, channel, out_h, out_w])) / area
					for in_h in start_h .. end_h {
						for in_w in start_w .. end_w {
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

pub fn adaptive_maxpool2d_forward[T](input &vtl.Tensor[T], output_size []int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool2d_output_shape[T](input, output_size)!
	mut output := vtl.zeros[T](output_shape)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_h in 0 .. output_size[0] {
				start_h := (out_h * input.shape[2]) / output_size[0]
				end_h := ((out_h + 1) * input.shape[2] + output_size[0] - 1) / output_size[0]
				for out_w in 0 .. output_size[1] {
					start_w := (out_w * input.shape[3]) / output_size[1]
					end_w := ((out_w + 1) * input.shape[3] + output_size[1] - 1) / output_size[1]
					mut maximum := f64(input.get([batch, channel, start_h, start_w]))
					for in_h in start_h .. end_h {
						for in_w in start_w .. end_w {
							value := f64(input.get([batch, channel, in_h, in_w]))
							if value > maximum {
								maximum = value
							}
						}
					}
					output.set([batch, channel, out_h, out_w], vtl.cast[T](maximum))
				}
			}
		}
	}
	return output
}

pub fn adaptive_maxpool2d_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T], output_size []int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool2d_output_shape[T](input, output_size)!
	if grad_out.shape != output_shape {
		return error('adaptive_maxpool2d backward gradient shape does not match input and output size')
	}
	mut d_input := vtl.zeros_like[T](input)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_h in 0 .. output_size[0] {
				start_h := (out_h * input.shape[2]) / output_size[0]
				end_h := ((out_h + 1) * input.shape[2] + output_size[0] - 1) / output_size[0]
				for out_w in 0 .. output_size[1] {
					start_w := (out_w * input.shape[3]) / output_size[1]
					end_w := ((out_w + 1) * input.shape[3] + output_size[1] - 1) / output_size[1]
					mut max_h := start_h
					mut max_w := start_w
					mut maximum := f64(input.get([batch, channel, start_h, start_w]))
					for in_h in start_h .. end_h {
						for in_w in start_w .. end_w {
							value := f64(input.get([batch, channel, in_h, in_w]))
							if value > maximum {
								maximum = value
								max_h = in_h
								max_w = in_w
							}
						}
					}
					index := [batch, channel, max_h, max_w]
					value := f64(d_input.get(index)) + f64(grad_out.get([batch, channel, out_h,
						out_w]))
					d_input.set(index, vtl.cast[T](value))
				}
			}
		}
	}
	return d_input
}

fn adaptive_avgpool1d_output_shape[T](input &vtl.Tensor[T], output_size int) ![]int {
	if input.rank() != 3 {
		return error('adaptive_avgpool1d expects a 3D input tensor')
	}
	if output_size <= 0 || input.shape[2] <= 0 {
		return error('adaptive_avgpool1d expects positive dimensions')
	}
	return [input.shape[0], input.shape[1], output_size]
}

pub fn adaptive_avgpool1d_forward[T](input &vtl.Tensor[T], output_size int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool1d_output_shape[T](input, output_size)!
	mut output := vtl.zeros[T](output_shape)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_i in 0 .. output_size {
				start := (out_i * input.shape[2]) / output_size
				end := ((out_i + 1) * input.shape[2] + output_size - 1) / output_size
				mut sum := f64(0)
				for in_i in start .. end {
					sum += f64(input.get([batch, channel, in_i]))
				}
				output.set([batch, channel, out_i], vtl.cast[T](sum / f64(end - start)))
			}
		}
	}
	return output
}

pub fn adaptive_avgpool1d_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T], output_size int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool1d_output_shape[T](input, output_size)!
	if grad_out.shape != output_shape {
		return error('adaptive_avgpool1d backward gradient shape does not match input and output size')
	}
	mut d_input := vtl.zeros_like[T](input)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_i in 0 .. output_size {
				start := (out_i * input.shape[2]) / output_size
				end := ((out_i + 1) * input.shape[2] + output_size - 1) / output_size
				contribution := f64(grad_out.get([batch, channel, out_i])) / f64(end - start)
				for in_i in start .. end {
					index := [batch, channel, in_i]
					value := f64(d_input.get(index)) + contribution
					d_input.set(index, vtl.cast[T](value))
				}
			}
		}
	}
	return d_input
}

pub fn adaptive_maxpool1d_forward[T](input &vtl.Tensor[T], output_size int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool1d_output_shape[T](input, output_size)!
	mut output := vtl.zeros[T](output_shape)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_i in 0 .. output_size {
				start := (out_i * input.shape[2]) / output_size
				end := ((out_i + 1) * input.shape[2] + output_size - 1) / output_size
				mut maximum := f64(input.get([batch, channel, start]))
				for in_i in start .. end {
					value := f64(input.get([batch, channel, in_i]))
					if value > maximum {
						maximum = value
					}
				}
				output.set([batch, channel, out_i], vtl.cast[T](maximum))
			}
		}
	}
	return output
}

pub fn adaptive_maxpool1d_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T], output_size int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool1d_output_shape[T](input, output_size)!
	if grad_out.shape != output_shape {
		return error('adaptive_maxpool1d backward gradient shape does not match input and output size')
	}
	mut d_input := vtl.zeros_like[T](input)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_i in 0 .. output_size {
				start := (out_i * input.shape[2]) / output_size
				end := ((out_i + 1) * input.shape[2] + output_size - 1) / output_size
				mut max_i := start
				mut maximum := f64(input.get([batch, channel, start]))
				for in_i in start .. end {
					value := f64(input.get([batch, channel, in_i]))
					if value > maximum {
						maximum = value
						max_i = in_i
					}
				}
				index := [batch, channel, max_i]
				value := f64(d_input.get(index)) + f64(grad_out.get([batch, channel, out_i]))
				d_input.set(index, vtl.cast[T](value))
			}
		}
	}
	return d_input
}

fn adaptive_avgpool3d_output_shape[T](input &vtl.Tensor[T], output_size []int) ![]int {
	if input.rank() != 5 {
		return error('adaptive_avgpool3d expects a 5D input tensor')
	}
	if output_size.len != 3 || output_size[0] <= 0 || output_size[1] <= 0 || output_size[2] <= 0 {
		return error('adaptive_avgpool3d expects three positive output dimensions')
	}
	if input.shape[2] <= 0 || input.shape[3] <= 0 || input.shape[4] <= 0 {
		return error('adaptive_avgpool3d expects positive spatial dimensions')
	}
	return [input.shape[0], input.shape[1], output_size[0], output_size[1], output_size[2]]
}

pub fn adaptive_avgpool3d_forward[T](input &vtl.Tensor[T], output_size []int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool3d_output_shape[T](input, output_size)!
	mut output := vtl.zeros[T](output_shape)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_d in 0 .. output_size[0] {
				start_d := (out_d * input.shape[2]) / output_size[0]
				end_d := ((out_d + 1) * input.shape[2] + output_size[0] - 1) / output_size[0]
				for out_h in 0 .. output_size[1] {
					start_h := (out_h * input.shape[3]) / output_size[1]
					end_h := ((out_h + 1) * input.shape[3] + output_size[1] - 1) / output_size[1]
					for out_w in 0 .. output_size[2] {
						start_w := (out_w * input.shape[4]) / output_size[2]
						end_w := ((out_w + 1) * input.shape[4] + output_size[2] - 1) / output_size[2]
						mut sum := f64(0)
						for in_d in start_d .. end_d {
							for in_h in start_h .. end_h {
								for in_w in start_w .. end_w {
									sum += f64(input.get([batch, channel, in_d, in_h, in_w]))
								}
							}
						}
						volume := (end_d - start_d) * (end_h - start_h) * (end_w - start_w)
						output.set([batch, channel, out_d, out_h, out_w], vtl.cast[T](sum / f64(volume)))
					}
				}
			}
		}
	}
	return output
}

pub fn adaptive_avgpool3d_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T], output_size []int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool3d_output_shape[T](input, output_size)!
	if grad_out.shape != output_shape {
		return error('adaptive_avgpool3d backward gradient shape does not match input and output size')
	}
	mut d_input := vtl.zeros_like[T](input)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_d in 0 .. output_size[0] {
				start_d := (out_d * input.shape[2]) / output_size[0]
				end_d := ((out_d + 1) * input.shape[2] + output_size[0] - 1) / output_size[0]
				for out_h in 0 .. output_size[1] {
					start_h := (out_h * input.shape[3]) / output_size[1]
					end_h := ((out_h + 1) * input.shape[3] + output_size[1] - 1) / output_size[1]
					for out_w in 0 .. output_size[2] {
						start_w := (out_w * input.shape[4]) / output_size[2]
						end_w := ((out_w + 1) * input.shape[4] + output_size[2] - 1) / output_size[2]
						volume := f64((end_d - start_d) * (end_h - start_h) * (end_w - start_w))
						contribution := f64(grad_out.get([batch, channel, out_d, out_h, out_w])) / volume
						for in_d in start_d .. end_d {
							for in_h in start_h .. end_h {
								for in_w in start_w .. end_w {
									index := [batch, channel, in_d, in_h, in_w]
									value := f64(d_input.get(index)) + contribution
									d_input.set(index, vtl.cast[T](value))
								}
							}
						}
					}
				}
			}
		}
	}
	return d_input
}

pub fn adaptive_maxpool3d_forward[T](input &vtl.Tensor[T], output_size []int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool3d_output_shape[T](input, output_size)!
	mut output := vtl.zeros[T](output_shape)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_d in 0 .. output_size[0] {
				start_d := (out_d * input.shape[2]) / output_size[0]
				end_d := ((out_d + 1) * input.shape[2] + output_size[0] - 1) / output_size[0]
				for out_h in 0 .. output_size[1] {
					start_h := (out_h * input.shape[3]) / output_size[1]
					end_h := ((out_h + 1) * input.shape[3] + output_size[1] - 1) / output_size[1]
					for out_w in 0 .. output_size[2] {
						start_w := (out_w * input.shape[4]) / output_size[2]
						end_w := ((out_w + 1) * input.shape[4] + output_size[2] - 1) / output_size[2]
						mut maximum := f64(input.get([batch, channel, start_d, start_h, start_w]))
						for in_d in start_d .. end_d {
							for in_h in start_h .. end_h {
								for in_w in start_w .. end_w {
									value := f64(input.get([batch, channel, in_d, in_h, in_w]))
									if value > maximum {
										maximum = value
									}
								}
							}
						}
						output.set([batch, channel, out_d, out_h, out_w], vtl.cast[T](maximum))
					}
				}
			}
		}
	}
	return output
}

pub fn adaptive_maxpool3d_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T], output_size []int) !&vtl.Tensor[T] {
	output_shape := adaptive_avgpool3d_output_shape[T](input, output_size)!
	if grad_out.shape != output_shape {
		return error('adaptive_maxpool3d backward gradient shape does not match input and output size')
	}
	mut d_input := vtl.zeros_like[T](input)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_d in 0 .. output_size[0] {
				start_d := (out_d * input.shape[2]) / output_size[0]
				end_d := ((out_d + 1) * input.shape[2] + output_size[0] - 1) / output_size[0]
				for out_h in 0 .. output_size[1] {
					start_h := (out_h * input.shape[3]) / output_size[1]
					end_h := ((out_h + 1) * input.shape[3] + output_size[1] - 1) / output_size[1]
					for out_w in 0 .. output_size[2] {
						start_w := (out_w * input.shape[4]) / output_size[2]
						end_w := ((out_w + 1) * input.shape[4] + output_size[2] - 1) / output_size[2]
						mut max_d := start_d
						mut max_h := start_h
						mut max_w := start_w
						mut maximum := f64(input.get([batch, channel, start_d, start_h, start_w]))
						for in_d in start_d .. end_d {
							for in_h in start_h .. end_h {
								for in_w in start_w .. end_w {
									value := f64(input.get([batch, channel, in_d, in_h, in_w]))
									if value > maximum {
										maximum = value
										max_d = in_d
										max_h = in_h
										max_w = in_w
									}
								}
							}
						}
						index := [batch, channel, max_d, max_h, max_w]
						value := f64(d_input.get(index)) + f64(grad_out.get([batch, channel, out_d,
							out_h, out_w]))
						d_input.set(index, vtl.cast[T](value))
					}
				}
			}
		}
	}
	return d_input
}

fn pool1d_output_shape[T](input &vtl.Tensor[T], kernel int, padding int, stride int) ![]int {
	if input.rank() != 3 {
		return error('avgpool1d expects a 3D input tensor')
	}
	if kernel <= 0 || stride <= 0 || padding < 0 {
		return error('avgpool1d expects positive kernel and stride and non-negative padding')
	}
	output_size := (input.shape[2] - kernel + 2 * padding) / stride + 1
	if output_size <= 0 {
		return error('avgpool1d parameters produce an empty output')
	}
	return [input.shape[0], input.shape[1], output_size]
}

pub fn avgpool1d_forward[T](input &vtl.Tensor[T], kernel int, padding int, stride int) !&vtl.Tensor[T] {
	output_shape := pool1d_output_shape[T](input, kernel, padding, stride)!
	mut output := vtl.zeros[T](output_shape)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_i in 0 .. output_shape[2] {
				start := out_i * stride - padding
				mut sum := f64(0)
				for offset in 0 .. kernel {
					index := start + offset
					if index >= 0 && index < input.shape[2] {
						sum += f64(input.get([batch, channel, index]))
					}
				}
				output.set([batch, channel, out_i], vtl.cast[T](sum / f64(kernel)))
			}
		}
	}
	return output
}

pub fn avgpool1d_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T], kernel int, padding int, stride int) !&vtl.Tensor[T] {
	output_shape := pool1d_output_shape[T](input, kernel, padding, stride)!
	if grad_out.shape != output_shape {
		return error('avgpool1d backward gradient shape does not match expected output')
	}
	mut d_input := vtl.zeros_like[T](input)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_i in 0 .. output_shape[2] {
				start := out_i * stride - padding
				contribution := f64(grad_out.get([batch, channel, out_i])) / f64(kernel)
				for offset in 0 .. kernel {
					index := start + offset
					if index >= 0 && index < input.shape[2] {
						position := [batch, channel, index]
						value := f64(d_input.get(position)) + contribution
						d_input.set(position, vtl.cast[T](value))
					}
				}
			}
		}
	}
	return d_input
}

pub fn maxpool1d_forward[T](input &vtl.Tensor[T], kernel int, padding int, stride int) !&vtl.Tensor[T] {
	output_shape := pool1d_output_shape[T](input, kernel, padding, stride)!
	mut output := vtl.zeros[T](output_shape)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_i in 0 .. output_shape[2] {
				start := out_i * stride - padding
				mut maximum := f64(0)
				mut found := false
				for offset in 0 .. kernel {
					index := start + offset
					if index >= 0 && index < input.shape[2] {
						value := f64(input.get([batch, channel, index]))
						if !found || value > maximum {
							maximum = value
							found = true
						}
					}
				}
				output.set([batch, channel, out_i], vtl.cast[T](maximum))
			}
		}
	}
	return output
}

pub fn maxpool1d_backward[T](grad_out &vtl.Tensor[T], input &vtl.Tensor[T], kernel int, padding int, stride int) !&vtl.Tensor[T] {
	output_shape := pool1d_output_shape[T](input, kernel, padding, stride)!
	if grad_out.shape != output_shape {
		return error('maxpool1d backward gradient shape does not match expected output')
	}
	mut d_input := vtl.zeros_like[T](input)
	for batch in 0 .. input.shape[0] {
		for channel in 0 .. input.shape[1] {
			for out_i in 0 .. output_shape[2] {
				start := out_i * stride - padding
				mut max_index := -1
				mut maximum := f64(0)
				for offset in 0 .. kernel {
					index := start + offset
					if index >= 0 && index < input.shape[2] {
						value := f64(input.get([batch, channel, index]))
						if max_index < 0 || value > maximum {
							maximum = value
							max_index = index
						}
					}
				}
				if max_index >= 0 {
					position := [batch, channel, max_index]
					value := f64(d_input.get(position)) + f64(grad_out.get([batch, channel, out_i]))
					d_input.set(position, vtl.cast[T](value))
				}
			}
		}
	}
	return d_input
}

fn pool3d_output_shape[T](input &vtl.Tensor[T], kernel []int, padding []int, stride []int) ![]int {
	if input.rank() != 5 || kernel.len != 3 || padding.len != 3 || stride.len != 3 {
		return error('avgpool3d expects a 5D input and three kernel, padding, and stride values')
	}
	for i in 0 .. 3 {
		if kernel[i] <= 0 || stride[i] <= 0 || padding[i] < 0 {
			return error('avgpool3d expects positive kernel and stride and non-negative padding')
		}
	}
	output := [
		(input.shape[2] - kernel[0] + 2 * padding[0]) / stride[0] + 1,
		(input.shape[3] - kernel[1] + 2 * padding[1]) / stride[1] + 1,
		(input.shape[4] - kernel[2] + 2 * padding[2]) / stride[2] + 1,
	]
	if output.any(it <= 0) {
		return error('avgpool3d parameters produce an empty output')
	}
	return [input.shape[0], input.shape[1], output[0], output[1], output[2]]
}

pub fn avgpool3d_forward[T](input &vtl.Tensor[T], kernel []int, padding []int, stride []int) !&vtl.Tensor[T] {
	output_shape := pool3d_output_shape[T](input, kernel, padding, stride)!
	mut output := vtl.zeros[T](output_shape)
	for b in 0 .. input.shape[0] {
		for c in 0 .. input.shape[1] {
			for od in 0 .. output_shape[2] {
				for oh in 0 .. output_shape[3] {
					for ow in 0 .. output_shape[4] {
						mut sum := f64(0)
						for kd in 0 .. kernel[0] {
							for kh in 0 .. kernel[1] {
								for kw in 0 .. kernel[2] {
									id := od * stride[0] - padding[0] + kd
									ih := oh * stride[1] - padding[1] + kh
									iw := ow * stride[2] - padding[2] + kw
									if id >= 0 && id < input.shape[2] && ih >= 0 && ih < input.shape[3] && iw >= 0 && iw < input.shape[4] {
										sum += f64(input.get([b, c, id, ih, iw]))
									}
								}
							}
						}
						volume := kernel[0] * kernel[1] * kernel[2]
						output.set([b, c, od, oh, ow], vtl.cast[T](sum / f64(volume)))
					}
				}
			}
		}
	}
	return output
}
