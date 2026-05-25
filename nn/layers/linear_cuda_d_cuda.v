module layers

import vtl
import vtl.storage
import vtl.la
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.layers
import vtl.nn.types
import vsl.cuda
import vsl.cuda.compute

// linear_forward_cuda computes y = x * W^T + b using cuBLAS GEMM
// x: [M, K] (input matrix)
// W: [N, K] (weights matrix)
// b: [N] or [1, N] (bias vector)
// Returns: [M, N] (output matrix)
pub fn linear_forward_cuda[T](x &Tensor[T], weights &Tensor[T], bias &Tensor[T]) !&Tensor[T] {
	assert x.is_matrix()
	assert weights.is_matrix()
	assert bias.is_vector() || (bias.is_matrix() && bias.shape[0] == 1)

	// Convert to CUDA tensors for GPU computation
	mut x_cuda := x.cuda()!
	mut w_cuda := weights.cuda()!

	// GEMM: x * W^T using cuBLAS - the gemm_cuda function expects f64 tensors
	// We need to get the underlying arrays and use compute directly
	dev := x_cuda.data.device
	x_arr := x_cuda.data.to_array()!
	w_arr := w_cuda.data.to_array()!

	m := x.shape[0]
	k := x.shape[1]
	n := weights.shape[0]

	// Convert row-major to column-major for cuBLAS
	x_col := cuda.row_to_col_major(x_arr, m, k)
	w_col := cuda.row_to_col_major(w_arr, n, k) // W is [N, K]

	// Run GEMM on GPU
	result_col := compute.gemm_cuda(dev, x_col, w_col, m, n, k)!

	// Convert result back to row-major
	gemm_result_row := cuda.col_to_row_major(result_col, m, n)

	// Add bias
	b_arr := bias.to_array()!
	mut final_result := gemm_result_row
	for i in 0 .. final_result.len {
		row_idx := i / n
		col_idx := i % n
		final_result[i] += b_arr[col_idx]
	}

	// Create result tensor as CudaStorage then convert to Tensor
	mut result_storage := &storage.CudaStorage[f64]{
		device: dev
	}
	mut ptr := voidptr(0)
	sz := int(sizeof(f64)) * final_result.len
	status := C.cudaMalloc(&ptr, sz)
	if status != 0 {
		return error('linear_forward_cuda: cudaMalloc failed with status ${status}')
	}
	result_storage.ptr = ptr
	result_storage.size = sz
	result_storage.count = final_result.len
	unsafe {
		C.cudaMemcpy(ptr, final_result.data, sz, C.cuda_memcpy_host_to_device)
	}

	x_cuda.release()
	w_cuda.release()

	return &Tensor[T]{
		data:    result_storage
		memory:  .row_major
		size:    final_result.len
		shape:   [m, n]
		strides: [n, 1]
	}
}

// relu_forward_cuda applies ReLU activation: max(0, x) on GPU
pub fn relu_forward_cuda[T](x &Tensor[T]) !&Tensor[T] {
	mut x_cuda := x.cuda()!
	dev := x_cuda.data.device
	input_data := x_cuda.data.to_array()!
	result := compute.relu_cuda(dev, input_data)!

	// Create output tensor on GPU
	mut output_storage := &storage.CudaStorage[f64]{
		device: dev
	}
	mut ptr := voidptr(0)
	sz := int(sizeof(f64)) * result.len
	status := C.cudaMalloc(&ptr, sz)
	if status != 0 {
		return error('relu_forward_cuda: cudaMalloc failed with status ${status}')
	}
	output_storage.ptr = ptr
	output_storage.size = sz
	output_storage.count = result.len
	unsafe {
		C.cudaMemcpy(ptr, result.data, sz, C.cuda_memcpy_host_to_device)
	}

	x_cuda.release()

	return &Tensor[T]{
		data:    output_storage
		memory:  x_cuda.memory
		size:    x_cuda.size
		shape:   x_cuda.shape
		strides: x_cuda.strides
	}
}

// sigmoid_forward_cuda applies Sigmoid activation: 1 / (1 + exp(-x)) on GPU
pub fn sigmoid_forward_cuda[T](x &Tensor[T]) !&Tensor[T] {
	mut x_cuda := x.cuda()!
	dev := x_cuda.data.device
	input_data := x_cuda.data.to_array()!
	result := compute.sigmoid_cuda(dev, input_data)!

	// Create output tensor on GPU
	mut output_storage := &storage.CudaStorage[f64]{
		device: dev
	}
	mut ptr := voidptr(0)
	sz := int(sizeof(f64)) * result.len
	status := C.cudaMalloc(&ptr, sz)
	if status != 0 {
		return error('sigmoid_forward_cuda: cudaMalloc failed with status ${status}')
	}
	output_storage.ptr = ptr
	output_storage.size = sz
	output_storage.count = result.len
	unsafe {
		C.cudaMemcpy(ptr, result.data, sz, C.cuda_memcpy_host_to_device)
	}

	x_cuda.release()

	return &Tensor[T]{
		data:    output_storage
		memory:  x_cuda.memory
		size:    x_cuda.size
		shape:   x_cuda.shape
		strides: x_cuda.strides
	}
}