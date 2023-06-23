module internal

import math
import vtl

pub fn dropout[T](input &vtl.Tensor[T], mask &vtl.Tensor[T], prob f64) !&vtl.Tensor[T] {
	mut iters, shape := input.iterators[T]([mask])!
	mut ret := vtl.tensor_like_with_shape[T](input, shape)
	for {
		vals, i := iters.next() or { break }
		val := vals[0] * vals[1] / vtl.cast[T](prob)
		ret.set(i, val)
	}
	return ret
}

pub fn dropout_backwards[T](gradient &vtl.Tensor[T], mask &vtl.Tensor[T], prob f64) !&vtl.Tensor[T] {
	mut iters, shape := gradient.iterators[T]([mask])
	mut ret := vtl.tensor_like_with_shape[T](gradient, shape)
	for {
		vals, i := iters.next() or { break }
		val := vals[0] * vals[1] / vtl.cast[T](prob)
		ret.set(i, val)
	}
	return ret
}

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

					mut max_val := vtl.cast[T](math.max_f64)
					mut max_idx := -1
					for i in hstart .. hend {
						for j in wstart .. wend {
							idx := i * ww + j
							val := input.get([n, c, i, j])
							if val > max_val {
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

pub fn maxpool2d_backward[T](shape []int, max_indices &vtl.Tensor[int], grad_output &vtl.Tensor[T]) &vtl.Tensor[T] {
	if grad_output.size != max_indices.size {
		panic('maxpool2d_backward: grad_output and max_indices must have the same size')
	}

	// TODO: @ulises-jeremias to override this on other backends
	mut ret := vtl.zeros[T](shape)
	for i in 0 .. grad_output.size {
		idx := max_indices.get_nth[int](i)
		ret.set_nth(idx, grad_output.get_nth(i))
	}
	return ret
}
