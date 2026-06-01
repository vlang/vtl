module layers

import vtl
import vtl.autograd_cuda
import vsl.cuda
import vsl.cuda.compute

// linear_forward_cuda_f64 computes y = x·Wᵀ + b using cuBLAS GEMM.
// Returns a CPU-resident tensor so autograd gates (CPU matmul) stay valid.
// Opt-in via VTL_USE_CUDA=1 and build flag `-d cuda`.
pub fn linear_forward_cuda_f64(x &vtl.Tensor[f64], weights &vtl.Tensor[f64], bias &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	if !cuda_linear_enabled() {
		return error('linear_forward_cuda_f64: set VTL_USE_CUDA=1 to enable')
	}
	mut session := autograd_cuda.new_device_session()
	session.init_device()
	return session.linear_forward_f64(x, weights, bias, unsafe { nil })
}

// relu_forward_cuda applies ReLU on GPU and returns a CPU tensor (f64 only).
pub fn relu_forward_cuda(x &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	if !cuda_linear_enabled() {
		return error('relu_forward_cuda: set VTL_USE_CUDA=1')
	}
	dev := cuda.get_default_device()!
	input_data := x.to_array()
	result := compute.relu_cuda(dev, input_data)!
	return vtl.from_array(result, x.shape.clone())!
}

// sigmoid_forward_cuda applies Sigmoid on GPU and returns a CPU tensor (f64 only).
pub fn sigmoid_forward_cuda(x &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	if !cuda_linear_enabled() {
		return error('sigmoid_forward_cuda: set VTL_USE_CUDA=1')
	}
	dev := cuda.get_default_device()!
	input_data := x.to_array()
	result := compute.sigmoid_cuda(dev, input_data)!
	return vtl.from_array(result, x.shape.clone())!
}
