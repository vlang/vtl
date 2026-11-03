module internal

import vsl.cuda
import vtl
import vtl.autograd_cuda

// maxpool2d_cudnn_eligible reports whether MaxPool2D backward may run on cuDNN.
//
// Limits, all checked here:
//   - CUDA backward is enabled at runtime.
//   - padding is 0. Padding is handled by the CPU path only.
//   - grad_output has the pooled output shape for input, kernel and stride.
//   - no pooling window holds two equal maxima. cuDNN's routing of ties is not
//     documented to match the CPU rule (first maximum wins), so tied windows use the CPU path.
fn maxpool2d_cudnn_eligible(input &vtl.Tensor[f64], max_indices &vtl.Tensor[int], grad_output &vtl.Tensor[f64], kernel []int, stride []int, padding []int) bool {
	if !autograd_cuda.cuda_backward_enabled() {
		return false
	}
	if input.shape.len != 4 || grad_output.shape.len != 4 || kernel.len != 2 || stride.len != 2
		|| padding.len != 2 {
		return false
	}
	if padding[0] != 0 || padding[1] != 0 || kernel[0] <= 0 || kernel[1] <= 0 || stride[0] <= 0
		|| stride[1] <= 0 {
		return false
	}
	n := input.shape[0]
	c := input.shape[1]
	h := input.shape[2]
	w := input.shape[3]
	if h < kernel[0] || w < kernel[1] {
		return false
	}
	out_h := (h - kernel[0]) / stride[0] + 1
	out_w := (w - kernel[1]) / stride[1] + 1
	if grad_output.shape != [n, c, out_h, out_w] || max_indices.shape != grad_output.shape {
		return false
	}

	x := input.to_array()
	for b in 0 .. n {
		for ch in 0 .. c {
			for oh in 0 .. out_h {
				for ow in 0 .. out_w {
					mut best := f64(0)
					mut best_count := 0
					for kh in 0 .. kernel[0] {
						for kw in 0 .. kernel[1] {
							value := x[((b * c + ch) * h + oh * stride[0] + kh) * w + ow * stride[1] + kw]
							if best_count == 0 || value > best {
								best = value
								best_count = 1
							} else if value == best {
								best_count++
							}
						}
					}
					if best_count > 1 {
						return false
					}
				}
			}
		}
	}
	return true
}

// maxpool2d_backward_cuda_f64 computes MaxPool2D input gradients with cudnnPoolingBackward.
// cuDNN routes gradients by comparing x with y, so y is gathered from input at max_indices.
// The caller must have checked maxpool2d_cudnn_eligible.
fn maxpool2d_backward_cuda_f64(shape []int, max_indices &vtl.Tensor[int], input &vtl.Tensor[f64], grad_output &vtl.Tensor[f64], kernel []int, stride []int) !&vtl.Tensor[f64] {
	n := shape[0]
	c := shape[1]
	h := shape[2]
	w := shape[3]
	out_h := grad_output.shape[2]
	out_w := grad_output.shape[3]
	in_count := n * c * h * w
	out_count := n * c * out_h * out_w

	x_values := input.to_array()
	idx_values := max_indices.to_array()
	mut y_values := []f64{len: out_count}
	for i, idx in idx_values {
		if idx < 0 || idx >= in_count {
			return error('maxpool2d cuDNN backward: max index ${idx} is out of bounds')
		}
		y_values[i] = x_values[idx]
	}

	mut dev := cuda.get_default_device()!
	defer {
		dev.release() or {}
	}

	x_desc := new_f64_nchw_desc(n, c, h, w)!
	defer {
		C.cudnnDestroyTensorDescriptor(x_desc)
	}
	y_desc := new_f64_nchw_desc(n, c, out_h, out_w)!
	defer {
		C.cudnnDestroyTensorDescriptor(y_desc)
	}
	dy_desc := new_f64_nchw_desc(n, c, out_h, out_w)!
	defer {
		C.cudnnDestroyTensorDescriptor(dy_desc)
	}
	dx_desc := new_f64_nchw_desc(n, c, h, w)!
	defer {
		C.cudnnDestroyTensorDescriptor(dx_desc)
	}

	mut pool_desc := CudnnPoolingDescriptor(unsafe { nil })
	cudnn_check(C.cudnnCreatePoolingDescriptor(&pool_desc), 'cudnnCreatePoolingDescriptor')!
	defer {
		C.cudnnDestroyPoolingDescriptor(pool_desc)
	}
	cudnn_check(C.cudnnSetPooling2dDescriptor(pool_desc, cudnn_pooling_mode_max, cudnn_nan_not_propagate,
		kernel[0], kernel[1], 0, 0, stride[0], stride[1]), 'cudnnSetPooling2dDescriptor')!

	d_x := device_alloc_bytes(in_count * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_x)
	}
	d_y := device_alloc_bytes(out_count * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_y)
	}
	d_dy := device_alloc_bytes(out_count * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dy)
	}
	d_dx := device_alloc_bytes(in_count * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dx)
	}

	device_upload_f64(d_x, x_values)!
	device_upload_f64(d_y, y_values)!
	device_upload_f64(d_dy, grad_output.to_array())!

	alpha := f64(1.0)
	beta := f64(0.0)
	cudnn_check(C.cudnnPoolingBackward(dev.cudnn, pool_desc, &alpha, y_desc, d_y, dy_desc, d_dy,
		x_desc, d_x, &beta, dx_desc, d_dx), 'cudnnPoolingBackward')!

	mut dx_values := []f64{len: in_count}
	device_download_f64(d_dx, mut dx_values)!
	return vtl.from_array(dx_values, shape)!
}
