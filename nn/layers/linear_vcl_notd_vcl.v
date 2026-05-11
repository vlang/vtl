module layers

import vtl
import vtl.storage

// Stubs when compiling without `-d vcl` (mirrors linear_vulkan_notd_vulkan.v pattern).

pub fn linear_forward_vcl[T](_ &vtl.Tensor[T], _ &vtl.Tensor[T], _ &vtl.Tensor[T], _ storage.VclStorageParams) !&vtl.Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vcl" to use VCL forward paths')
}

pub fn relu_forward_vcl[T](_ &vtl.Tensor[T], _ storage.VclStorageParams) !&vtl.Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vcl" to use VCL forward paths')
}

pub fn sigmoid_forward_vcl[T](_ &vtl.Tensor[T], _ storage.VclStorageParams) !&vtl.Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vcl" to use VCL forward paths')
}
