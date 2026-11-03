module internal

import math
import vsl.cuda
import vsl.cuda.gpu
import vtl
import vtl.autograd_cuda

// batchnorm2d_cudnn_eligible reports whether training-mode BatchNorm2D backward may run on cuDNN.
// cuDNN requires epsilon >= CUDNN_BN_MIN_EPSILON; evaluation mode has no cuDNN path.
fn batchnorm2d_cudnn_eligible(eps f64, training bool) bool {
	return training && eps >= cudnn_bn_min_epsilon && autograd_cuda.cuda_backward_enabled()
}

// batchnorm2d_backward_cuda_f64 computes training-mode BatchNorm2D gradients with
// cudnnBatchNormalizationBackwardEx (spatial mode). mean and var_ are the biased batch
// statistics from batchnorm2d_training. cuDNN takes the inverse standard deviation,
// so it is derived on the host from var_.
fn batchnorm2d_backward_cuda_f64(gradient &vtl.Tensor[f64], input &vtl.Tensor[f64], gamma &vtl.Tensor[f64], mean &vtl.Tensor[f64], var_ &vtl.Tensor[f64], eps f64) ![]&vtl.Tensor[f64] {
	if input.shape.len != 4 {
		return error('batchnorm2d cuDNN backward: input must be 4D NCHW')
	}
	if gradient.shape != input.shape {
		return error('batchnorm2d cuDNN backward: gradient shape must match input shape')
	}
	if eps < cudnn_bn_min_epsilon {
		return error('batchnorm2d cuDNN backward: epsilon ${eps} is below CUDNN_BN_MIN_EPSILON')
	}
	n := input.shape[0]
	c := input.shape[1]
	h := input.shape[2]
	w := input.shape[3]
	count := n * c * h * w

	mut inv_std := []f64{len: c}
	for ch in 0 .. c {
		inv_std[ch] = 1.0 / math.sqrt(var_.get([ch]) + eps)
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

	mut d_x := gpu.cuda_memory_new[f64](count)!
	defer {
		d_x.free() or {}
	}
	mut d_y := gpu.cuda_memory_new[f64](count)!
	defer {
		d_y.free() or {}
	}
	mut d_dy := gpu.cuda_memory_new[f64](count)!
	defer {
		d_dy.free() or {}
	}
	mut d_dx := gpu.cuda_memory_new[f64](count)!
	defer {
		d_dx.free() or {}
	}
	mut d_gamma := gpu.cuda_memory_new[f64](c)!
	defer {
		d_gamma.free() or {}
	}
	mut d_mean := gpu.cuda_memory_new[f64](c)!
	defer {
		d_mean.free() or {}
	}
	mut d_inv_std := gpu.cuda_memory_new[f64](c)!
	defer {
		d_inv_std.free() or {}
	}
	mut d_dgamma := gpu.cuda_memory_new[f64](c)!
	defer {
		d_dgamma.free() or {}
	}
	mut d_dbeta := gpu.cuda_memory_new[f64](c)!
	defer {
		d_dbeta.free() or {}
	}

	mut ws := gpu.CudaMemory[u8]{}
	mut ws_ptr := unsafe { voidptr(nil) }
	if ws_bytes > 0 {
		ws = gpu.cuda_memory_new[u8](int(ws_bytes))!
		ws_ptr = ws.ptr
	}
	defer {
		ws.free() or {}
	}

	d_x.upload(input.to_array())!
	d_dy.upload(gradient.to_array())!
	d_gamma.upload(gamma.to_array())!
	d_mean.upload(mean.to_array())!
	d_inv_std.upload(inv_std)!

	alpha := f64(1.0)
	beta := f64(0.0)
	cudnn_check(C.cudnnBatchNormalizationBackwardEx(dev.cudnn, cudnn_bn_mode_spatial, cudnn_bn_ops_bn,
		&alpha, &beta, &alpha, &beta, x_desc, d_x.ptr, y_desc, d_y.ptr, dy_desc, d_dy.ptr,
		unsafe { nil }, unsafe { nil }, dx_desc, d_dx.ptr, bn_desc, d_gamma.ptr, unsafe { nil },
		d_dgamma.ptr, d_dbeta.ptr, eps, d_mean.ptr, d_inv_std.ptr, unsafe { nil }, ws_ptr,
		ws_bytes, unsafe { nil }, 0), 'cudnnBatchNormalizationBackwardEx')!

	mut dx_values := []f64{len: count}
	d_dx.download(mut dx_values)!
	mut dgamma_values := []f64{len: c}
	d_dgamma.download(mut dgamma_values)!
	mut dbeta_values := []f64{len: c}
	d_dbeta.download(mut dbeta_values)!

	dx := vtl.from_array(dx_values, input.shape)!
	dgamma := vtl.from_array(dgamma_values, [c])!
	dbeta := vtl.from_array(dbeta_values, [c])!
	return [dx, dgamma, dbeta]
}
