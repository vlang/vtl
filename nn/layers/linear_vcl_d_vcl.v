module layers

import vtl
import vtl.storage

// linear_forward_vcl, relu_forward_vcl, sigmoid_forward_vcl — VCL (OpenCL) compute backend.
//
// NOTE: VSL's vcl module is a transport layer (data transfer + kernel launch) but does NOT
// provide pre-built compute kernels (gemm/relu/sigmoid) like vsl.vulkan does.
// VCL supports arbitrary OpenCL kernel execution via device.add_program() but requires
// user-provided kernel source strings.
//
// Real VCL compute implementation (gemm on GPU via OpenCL) is a Phase 5 task (vtl#62).
// For now these return errors when compiled with -d vcl.

pub fn linear_forward_vcl[T](_ &vtl.Tensor[T], _ &vtl.Tensor[T], _ &vtl.Tensor[T], _ storage.VclStorageParams) !&vtl.Tensor[T] {
	return error(@METHOD + ':' + ' VCL compute (GEMM on GPU via OpenCL) is not yet implemented. ' +
		' Use CPU backend or Vulkan compute backend (-d vulkan). ' + ' See vtl#62 (Phase 5: OpenCL Backend)')
}

pub fn relu_forward_vcl[T](_ &vtl.Tensor[T], _ storage.VclStorageParams) !&vtl.Tensor[T] {
	return error(@METHOD + ':' + ' VCL compute (ReLU on GPU via OpenCL) is not yet implemented. ' +
		' Use CPU backend or Vulkan compute backend (-d vulkan). ' + ' See vtl#62 (Phase 5: OpenCL Backend)')
}

pub fn sigmoid_forward_vcl[T](_ &vtl.Tensor[T], _ storage.VclStorageParams) !&vtl.Tensor[T] {
	return error(@METHOD + ':' +
		' VCL compute (Sigmoid on GPU via OpenCL) is not yet implemented. ' +
		' Use CPU backend or Vulkan compute backend (-d vulkan). ' + ' See vtl#62 (Phase 5: OpenCL Backend)')
}
