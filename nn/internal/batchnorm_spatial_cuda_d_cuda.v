module internal

import math
import vsl.cuda
import vtl
import vtl.autograd_cuda

// batchnorm_cudnn_eligible reports whether training-mode BatchNorm backward may run on cuDNN.
// cuDNN requires epsilon >= CUDNN_BN_MIN_EPSILON; evaluation mode has no cuDNN path.
// The same gate serves BatchNorm1D, 2D and 3D.
fn batchnorm_cudnn_eligible(eps f64, training bool) bool {
	return training && eps >= cudnn_bn_min_epsilon && autograd_cuda.cuda_backward_enabled()
}

// batchnorm_spatial_backward_cuda_f64 computes training-mode BatchNorm gradients with
// cudnnBatchNormalizationBackwardEx in spatial mode over an NCHW view of the data:
// n samples, c channels, h x w positions. mean and var_ are the biased batch statistics.
// cuDNN takes the inverse standard deviation, so it is derived on the host from var_.
fn batchnorm_spatial_backward_cuda_f64(gradient []f64, input []f64, gamma []f64, mean []f64, var_ []f64, n int, c int, h int, w int, eps f64) !([]f64, []f64, []f64) {
	if eps < cudnn_bn_min_epsilon {
		return error('batchnorm cuDNN backward: epsilon ${eps} is below CUDNN_BN_MIN_EPSILON')
	}
	count := n * c * h * w

	mut inv_std := []f64{len: c}
	for ch in 0 .. c {
		inv_std[ch] = 1.0 / math.sqrt(var_[ch] + eps)
	}

	mut dev := cuda.get_default_device()!
	defer {
		dev.release() or {}
	}

	x_desc := new_f64_nchw_desc(n, c, h, w)!
	defer {
		C.cudnnDestroyTensorDescriptor(x_desc)
	}
	// y is a forward output that BatchNorm-only backward does not read.
	y_desc := new_f64_nchw_desc(n, c, h, w)!
	defer {
		C.cudnnDestroyTensorDescriptor(y_desc)
	}
	dy_desc := new_f64_nchw_desc(n, c, h, w)!
	defer {
		C.cudnnDestroyTensorDescriptor(dy_desc)
	}
	dx_desc := new_f64_nchw_desc(n, c, h, w)!
	defer {
		C.cudnnDestroyTensorDescriptor(dx_desc)
	}

	mut bn_desc := cuda.CudnnTensorDescriptor(unsafe { nil })
	cudnn_check(C.cudnnCreateTensorDescriptor(&bn_desc), 'cudnnCreateTensorDescriptor(bn)')!
	defer {
		C.cudnnDestroyTensorDescriptor(bn_desc)
	}
	cudnn_check(C.cudnnDeriveBNTensorDescriptor(bn_desc, x_desc, cudnn_bn_mode_spatial),
		'cudnnDeriveBNTensorDescriptor')!

	mut ws_bytes := usize(0)
	cudnn_check(C.cudnnGetBatchNormalizationBackwardExWorkspaceSize(dev.cudnn, cudnn_bn_mode_spatial,
		cudnn_bn_ops_bn, x_desc, y_desc, dy_desc, unsafe { nil }, dx_desc, bn_desc, unsafe { nil },
		&ws_bytes), 'cudnnGetBatchNormalizationBackwardExWorkspaceSize')!

	d_x := device_alloc_bytes(count * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_x)
	}
	d_y := device_alloc_bytes(count * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_y)
	}
	d_dy := device_alloc_bytes(count * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dy)
	}
	d_dx := device_alloc_bytes(count * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dx)
	}
	d_gamma := device_alloc_bytes(c * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_gamma)
	}
	d_mean := device_alloc_bytes(c * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_mean)
	}
	d_inv_std := device_alloc_bytes(c * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_inv_std)
	}
	d_dgamma := device_alloc_bytes(c * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dgamma)
	}
	d_dbeta := device_alloc_bytes(c * int(sizeof(f64)))!
	defer {
		C.cudaFree(d_dbeta)
	}

	mut ws_ptr := unsafe { voidptr(nil) }
	if ws_bytes > 0 {
		ws_ptr = device_alloc_bytes(int(ws_bytes))!
	}
	defer {
		if ws_ptr != unsafe { nil } {
			C.cudaFree(ws_ptr)
		}
	}

	device_upload_f64(d_x, input)!
	device_upload_f64(d_dy, gradient)!
	device_upload_f64(d_gamma, gamma)!
	device_upload_f64(d_mean, mean)!
	device_upload_f64(d_inv_std, inv_std)!

	alpha := f64(1.0)
	beta := f64(0.0)
	cudnn_check(C.cudnnBatchNormalizationBackwardEx(dev.cudnn, cudnn_bn_mode_spatial, cudnn_bn_ops_bn,
		&alpha, &beta, &alpha, &beta, x_desc, d_x, y_desc, d_y, dy_desc, d_dy, unsafe { nil },
		unsafe { nil }, dx_desc, d_dx, bn_desc, d_gamma, unsafe { nil }, d_dgamma, d_dbeta, eps,
		d_mean, d_inv_std, unsafe { nil }, ws_ptr, ws_bytes, unsafe { nil }, 0),
		'cudnnBatchNormalizationBackwardEx')!

	mut dx_values := []f64{len: count}
	device_download_f64(d_dx, mut dx_values)!
	mut dgamma_values := []f64{len: c}
	device_download_f64(d_dgamma, mut dgamma_values)!
	mut dbeta_values := []f64{len: c}
	device_download_f64(d_dbeta, mut dbeta_values)!
	return dx_values, dgamma_values, dbeta_values
}

// batchnorm2d_backward_cuda_f64 computes training-mode BatchNorm2D gradients on cuDNN.
// dgamma and dbeta have shape [C], matching the CPU path.
fn batchnorm2d_backward_cuda_f64(gradient &vtl.Tensor[f64], input &vtl.Tensor[f64], gamma &vtl.Tensor[f64], mean &vtl.Tensor[f64], var_ &vtl.Tensor[f64], eps f64) ![]&vtl.Tensor[f64] {
	if input.shape.len != 4 {
		return error('batchnorm2d cuDNN backward: input must be 4D NCHW')
	}
	if gradient.shape != input.shape {
		return error('batchnorm2d cuDNN backward: gradient shape must match input shape')
	}
	n := input.shape[0]
	c := input.shape[1]
	dx_values, dgamma_values, dbeta_values := batchnorm_spatial_backward_cuda_f64(gradient.to_array(),
		input.to_array(), gamma.to_array(), mean.to_array(), var_.to_array(), n, c, input.shape[2],
		input.shape[3], eps)!
	return [
		vtl.from_array(dx_values, input.shape)!,
		vtl.from_array(dgamma_values, [c])!,
		vtl.from_array(dbeta_values, [c])!,
	]
}

// batchnorm1d_backward_cuda_f64 computes training-mode BatchNorm1D gradients on cuDNN.
// A [N, C] input is the NCHW view (N, C, 1, 1). dgamma and dbeta have shape [1, C],
// matching the CPU path.
fn batchnorm1d_backward_cuda_f64(gradient &vtl.Tensor[f64], input &vtl.Tensor[f64], gamma &vtl.Tensor[f64], mean &vtl.Tensor[f64], var_ &vtl.Tensor[f64], eps f64) ![]&vtl.Tensor[f64] {
	if input.shape.len != 2 {
		return error('batchnorm1d cuDNN backward: input must be 2D [N, C]')
	}
	if gradient.shape != input.shape {
		return error('batchnorm1d cuDNN backward: gradient shape must match input shape')
	}
	n := input.shape[0]
	c := input.shape[1]
	dx_values, dgamma_values, dbeta_values := batchnorm_spatial_backward_cuda_f64(gradient.to_array(),
		input.to_array(), gamma.to_array(), mean.to_array(), var_.to_array(), n, c, 1, 1, eps)!
	return [
		vtl.from_array(dx_values, input.shape)!,
		vtl.from_array(dgamma_values, [1, c])!,
		vtl.from_array(dbeta_values, [1, c])!,
	]
}

// batchnorm3d_backward_cuda_f64 computes training-mode BatchNorm3D gradients on cuDNN.
// Spatial BatchNorm reduces over N and every position, so the depth and height axes merge
// into one (N, C, D*H, W) view. dgamma and dbeta have shape [C], matching the CPU path.
fn batchnorm3d_backward_cuda_f64(gradient &vtl.Tensor[f64], input &vtl.Tensor[f64], gamma &vtl.Tensor[f64], mean &vtl.Tensor[f64], var_ &vtl.Tensor[f64], eps f64) ![]&vtl.Tensor[f64] {
	if input.shape.len != 5 {
		return error('batchnorm3d cuDNN backward: input must be 5D NCDHW')
	}
	if gradient.shape != input.shape {
		return error('batchnorm3d cuDNN backward: gradient shape must match input shape')
	}
	n := input.shape[0]
	c := input.shape[1]
	dx_values, dgamma_values, dbeta_values := batchnorm_spatial_backward_cuda_f64(gradient.to_array(),
		input.to_array(), gamma.to_array(), mean.to_array(), var_.to_array(), n, c,
		input.shape[2] * input.shape[3], input.shape[4], eps)!
	return [
		vtl.from_array(dx_values, input.shape)!,
		vtl.from_array(dgamma_values, [c])!,
		vtl.from_array(dbeta_values, [c])!,
	]
}
