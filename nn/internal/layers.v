module internal

import math
import vtl

// dropout exposes this operation as part of the public API.
pub fn dropout[T](input &vtl.Tensor[T], mask &vtl.Tensor[T], prob f64) !&vtl.Tensor[T] {
	if prob < 0 || prob > 1 {
		return error('dropout: keep probability must be in [0, 1]')
	}
	if prob == 0 {
		return vtl.zeros_like[T](input)
	}
	return input.nmap([mask], fn [prob] [T](vals []T, i []int) T {
		return vals[0] * vals[1] / vtl.cast[T](prob)
	})
}

// dropout_backwards exposes this operation as part of the public API.
pub fn dropout_backwards[T](gradient &vtl.Tensor[T], mask &vtl.Tensor[T], prob f64) !&vtl.Tensor[T] {
	if prob < 0 || prob > 1 {
		return error('dropout backward: keep probability must be in [0, 1]')
	}
	if prob == 0 {
		return vtl.zeros_like[T](gradient)
	}
	return gradient.nmap([mask], fn [prob] [T](vals []T, i []int) T {
		return vals[0] * vals[1] / vtl.cast[T](prob)
	})
}

// maxpool2d exposes this operation as part of the public API.
pub fn maxpool2d[T](input &vtl.Tensor[T], kernel []int, padding []int, stride []int) (&vtl.Tensor[int], &vtl.Tensor[T]) {
	nn := input.shape[0]
	cc := input.shape[1]
	hh := input.shape[2]
	ww := input.shape[3]

	kk := kernel[0]
	kw := kernel[1]

	outh := (hh + 2 * padding[0] - kk) / stride[0] + 1
	outw := (ww + 2 * padding[1] - kw) / stride[1] + 1

	mut max_indices := vtl.zeros[int]([nn, cc, outh, outw])
	mut output := vtl.zeros[T]([nn, cc, outh, outw])

	for n in 0 .. nn {
		for c in 0 .. cc {
			for h in 0 .. outh {
				for w in 0 .. outw {
					mut hstart := h * stride[0] - padding[0]
					mut hend := hstart + kk
					mut wstart := w * stride[1] - padding[1]
					mut wend := wstart + kw

					hstart = math.max(hstart, 0)
					wstart = math.max(wstart, 0)
					hend = math.min(hend, hh)
					wend = math.min(wend, ww)

					mut max_val := vtl.cast[T](0)
					mut max_idx := -1
					for i in hstart .. hend {
						for j in wstart .. wend {
							idx := ((n * cc + c) * hh + i) * ww + j
							val := input.get([n, c, i, j])
							if max_idx < 0 || val > max_val {
								max_val = val
								max_idx = idx
							}
						}
					}
					output.set([n, c, h, w], max_val)
					max_indices.set([n, c, h, w], max_idx)
				}
			}
		}
	}

	return max_indices, output
}

// maxpool2d_backward_with_input routes the gradient through cuDNN when
// maxpool2d_cudnn_eligible allows it (`-d cuda`, VTL_USE_CUDA=1, VTL_CUDA_BACKWARD=1).
// Otherwise it runs the CPU scatter in maxpool2d_backward.
pub fn maxpool2d_backward_with_input[T](shape []int, max_indices &vtl.Tensor[int], input &vtl.Tensor[T], grad_output &vtl.Tensor[T], kernel []int, stride []int, padding []int) !&vtl.Tensor[T] {
	$if T is f64 {
		if maxpool2d_cudnn_eligible(input, max_indices, grad_output, kernel, stride, padding) {
			return maxpool2d_backward_cuda_f64(shape, max_indices, input, grad_output, kernel,
				stride)
		}
	}
	return maxpool2d_backward[T](shape, max_indices, grad_output)
}

// maxpool2d_backward exposes this operation as part of the public API.
pub fn maxpool2d_backward[T](shape []int, max_indices &vtl.Tensor[int], grad_output &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if grad_output.size != max_indices.size {
		return error('maxpool2d_backward: grad_output and max_indices must have the same size')
	}

	mut ret := vtl.zeros[T](shape)
	for i in 0 .. grad_output.size {
		idx := max_indices.get_nth[int](i)
		if idx < 0 || idx >= ret.size {
			return error('maxpool2d_backward: max index ${idx} is out of bounds')
		}
		ret.set_nth(idx, ret.get_nth(idx) + grad_output.get_nth(i))
	}
	return ret
}

pub fn maxpool3d[T](input &vtl.Tensor[T], kernel []int, padding []int, stride []int) !(&vtl.Tensor[int], &vtl.Tensor[T]) {
	if input.shape.len != 5 || kernel.len != 3 || padding.len != 3 || stride.len != 3 {
		return error('maxpool3d expects a 5D input and three kernel, padding, and stride values')
	}
	for value in kernel {
		if value <= 0 {
			return error('maxpool3d kernel values must be positive')
		}
	}
	for value in stride {
		if value <= 0 {
			return error('maxpool3d stride values must be positive')
		}
	}
	for value in padding {
		if value < 0 {
			return error('maxpool3d padding values must be non-negative')
		}
	}
	for axis in 0 .. 3 {
		if input.shape[axis + 2] + 2 * padding[axis] < kernel[axis] {
			return error('maxpool3d kernel does not fit the padded input')
		}
	}
	nn := input.shape[0]
	cc := input.shape[1]
	dd := input.shape[2]
	hh := input.shape[3]
	ww := input.shape[4]
	outd := (dd + 2 * padding[0] - kernel[0]) / stride[0] + 1
	outh := (hh + 2 * padding[1] - kernel[1]) / stride[1] + 1
	outw := (ww + 2 * padding[2] - kernel[2]) / stride[2] + 1
	mut max_indices := vtl.zeros[int]([nn, cc, outd, outh, outw])
	mut output := vtl.zeros[T]([nn, cc, outd, outh, outw])
	for n in 0 .. nn {
		for c in 0 .. cc {
			for od in 0 .. outd {
				for oh in 0 .. outh {
					for ow in 0 .. outw {
						dstart := od * stride[0] - padding[0]
						hstart := oh * stride[1] - padding[1]
						wstart := ow * stride[2] - padding[2]
						mut max_idx := -1
						mut max_val := vtl.cast[T](0)
						for kd in 0 .. kernel[0] {
							for kh in 0 .. kernel[1] {
								for kw in 0 .. kernel[2] {
									id, ih, iw := dstart + kd, hstart + kh, wstart + kw
									if id >= 0 && id < dd && ih >= 0 && ih < hh && iw >= 0 && iw < ww {
										idx := ((((n * cc + c) * dd + id) * hh + ih) * ww + iw)
										value := input.get([n, c, id, ih, iw])
										if max_idx < 0 || value > max_val {
											max_val = value
											max_idx = idx
										}
									}
								}
							}
						}
						output.set([n, c, od, oh, ow], max_val)
						max_indices.set([n, c, od, oh, ow], max_idx)
					}
				}
			}
		}
	}
	return max_indices, output
}

pub fn maxpool3d_backward[T](shape []int, max_indices &vtl.Tensor[int], grad_output &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if grad_output.size != max_indices.size {
		return error('maxpool3d_backward: grad_output and max_indices must have the same size')
	}
	mut ret := vtl.zeros[T](shape)
	for i in 0 .. grad_output.size {
		idx := max_indices.get_nth[int](i)
		if idx < 0 || idx >= ret.size {
			return error('maxpool3d_backward: max index ${idx} is out of bounds')
		}
		ret.set_nth(idx, ret.get_nth(idx) + grad_output.get_nth(i))
	}
	return ret
}
