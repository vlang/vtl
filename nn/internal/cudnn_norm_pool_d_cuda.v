module internal

import vsl.cuda

// cuDNN declarations for the GPU backward paths of BatchNorm2D and MaxPool2D
// (ADR-002, phase 1). vsl/cuda binds only the forward and convolution entry
// points, so these declarations live in VTL and vsl stays untouched.
//
// The cuDNN 9.27 entry points used here are marked CUDNN_DEPRECATED in the
// headers (cudnn_ops_v9.h) in favour of the graph API. They are still exported.

// Enum values from cudnn_ops_v9.h and cudnn_graph_v9.h (cuDNN 9.27).
// CUDNN_TENSOR_NCHW is 0 in cuDNN. vsl's cuda.cudnn_tensor_nchw is 1 (NHWC),
// so it must not be used for NCHW descriptors (see cudnn_graph_v9.h:390).
const cudnn_tensor_format_nchw = 0
const cudnn_bn_mode_spatial = 1
const cudnn_bn_ops_bn = 0
const cudnn_pooling_mode_max = 0
const cudnn_nan_not_propagate = 0

// CUDNN_BN_MIN_EPSILON from cudnn_ops_v9.h. cuDNN rejects smaller epsilons.
const cudnn_bn_min_epsilon = 1e-5

// CudnnPoolingDescriptor is an opaque cudnnPoolingDescriptor_t handle.
type CudnnPoolingDescriptor = voidptr

fn C.cudnnDeriveBNTensorDescriptor(derived_bn_desc cuda.CudnnTensorDescriptor, x_desc cuda.CudnnTensorDescriptor, mode int) int

fn C.cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle voidptr, mode int, bn_ops int, x_desc cuda.CudnnTensorDescriptor, y_desc cuda.CudnnTensorDescriptor, dy_desc cuda.CudnnTensorDescriptor, dz_desc cuda.CudnnTensorDescriptor, dx_desc cuda.CudnnTensorDescriptor, d_bn_scale_bias_desc cuda.CudnnTensorDescriptor, activation_desc voidptr, size_in_bytes &usize) int

fn C.cudnnBatchNormalizationBackwardEx(handle voidptr, mode int, bn_ops int, alpha_data_diff &f64, beta_data_diff &f64, alpha_param_diff &f64, beta_param_diff &f64, x_desc cuda.CudnnTensorDescriptor, x_data voidptr, y_desc cuda.CudnnTensorDescriptor, y_data voidptr, dy_desc cuda.CudnnTensorDescriptor, dy_data voidptr, dz_desc cuda.CudnnTensorDescriptor, dz_data voidptr, dx_desc cuda.CudnnTensorDescriptor, dx_data voidptr, d_bn_scale_bias_desc cuda.CudnnTensorDescriptor, bn_scale_data voidptr, bn_bias_data voidptr, d_bn_scale_data voidptr, d_bn_bias_data voidptr, epsilon f64, saved_mean voidptr, saved_inv_variance voidptr, activation_desc voidptr, workspace voidptr, workspace_size usize, reserve_space voidptr, reserve_space_size usize) int

fn C.cudnnCreatePoolingDescriptor(pooling_desc &CudnnPoolingDescriptor) int

fn C.cudnnSetPooling2dDescriptor(pooling_desc CudnnPoolingDescriptor, mode int, max_pooling_nan_opt int, window_height int, window_width int, vertical_padding int, horizontal_padding int, vertical_stride int, horizontal_stride int) int

fn C.cudnnDestroyPoolingDescriptor(pooling_desc CudnnPoolingDescriptor) int

fn C.cudnnPoolingBackward(handle voidptr, pooling_desc CudnnPoolingDescriptor, alpha &f64, y_desc cuda.CudnnTensorDescriptor, y voidptr, dy_desc cuda.CudnnTensorDescriptor, dy voidptr, x_desc cuda.CudnnTensorDescriptor, x voidptr, beta &f64, dx_desc cuda.CudnnTensorDescriptor, dx voidptr) int

// cudnn_check converts a cuDNN status into a V error that names the failing call.
fn cudnn_check(status int, call string) ! {
	if status != cuda.cudnn_status_success {
		return error('${call}: ${cuda.cudnn_error(status)}')
	}
}

// new_f64_nchw_desc creates an NCHW float64 tensor descriptor. The caller must destroy it.
fn new_f64_nchw_desc(n int, c int, h int, w int) !cuda.CudnnTensorDescriptor {
	mut desc := cuda.CudnnTensorDescriptor(unsafe { nil })
	cudnn_check(C.cudnnCreateTensorDescriptor(&desc), 'cudnnCreateTensorDescriptor')!
	cudnn_check(C.cudnnSetTensor4dDescriptor(desc, cudnn_tensor_format_nchw, cuda.cudnn_data_type_double,
		n, c, h, w), 'cudnnSetTensor4dDescriptor') or {
		C.cudnnDestroyTensorDescriptor(desc)
		return err
	}
	return desc
}

// device_alloc_bytes allocates device memory. The caller releases it with C.cudaFree.
// Raw runtime calls are used instead of vsl's generic gpu.CudaMemory, which broke
// codegen for other VTL CUDA tests when instantiated from this module.
fn device_alloc_bytes(bytes int) !voidptr {
	mut ptr := unsafe { voidptr(nil) }
	status := C.cudaMalloc(&ptr, bytes)
	if status != 0 {
		return error('cudaMalloc(${bytes} bytes) failed with status ${status}')
	}
	return ptr
}

// device_upload_f64 copies host float64 values to device memory at dst.
fn device_upload_f64(dst voidptr, src []f64) ! {
	if src.len == 0 {
		return
	}
	status := C.cudaMemcpy(dst, src.data, int(sizeof(f64)) * src.len, cuda.cuda_memcpy_host_to_device)
	if status != 0 {
		return error('cudaMemcpy host-to-device failed with status ${status}')
	}
}

// device_download_f64 copies device float64 values from src into dst.
fn device_download_f64(src voidptr, mut dst []f64) ! {
	if dst.len == 0 {
		return
	}
	status := C.cudaMemcpy(dst.data, src, int(sizeof(f64)) * dst.len, cuda.cuda_memcpy_device_to_host)
	if status != 0 {
		return error('cudaMemcpy device-to-host failed with status ${status}')
	}
}
