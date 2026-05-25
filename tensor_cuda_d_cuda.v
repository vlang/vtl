module vtl

import storage
import vsl.cuda
import vsl.cuda.compute

// CudaTensor holds tensor data on GPU memory
@[heap]
pub struct CudaTensor[T] {
pub mut:
	data    &storage.CudaStorage[T] = unsafe { nil }
	memory  MemoryFormat
	size    int
	shape   []int
	strides []int
}

// cuda creates a CudaTensor from a Tensor by copying data to GPU
pub fn (t &Tensor[T]) cuda(params storage.CudaParams) !&CudaTensor[T] {
	row_tensor := t.copy(.row_major)
	cudata := row_tensor.data.cuda(params)!
	return &CudaTensor[T]{
		data:    cudata
		memory:  row_tensor.memory
		size:    row_tensor.size
		shape:   row_tensor.shape
		strides: row_tensor.strides
	}
}

// cpu creates a Tensor from a CudaTensor by copying data from GPU to CPU
pub fn (t &CudaTensor[T]) cpu() !&Tensor[T] {
	data := t.data.cpu()!
	return &Tensor[T]{
		data:    data
		memory:  t.memory
		size:    t.size
		shape:   t.shape
		strides: t.strides
	}
}

// cuda returns the same CudaTensor (identity function for chaining)
@[inline]
pub fn (t &CudaTensor[T]) cuda(params storage.CudaParams) !&CudaTensor[T] {
	return t
}

// release releases the GPU memory
pub fn (t &CudaTensor[T]) release() {
	t.data.release()
}

// rank returns the number of dimensions
pub fn (t &CudaTensor[T]) rank() int {
	return t.shape.len
}

// numel returns the total number of elements
pub fn (t &CudaTensor[T]) numel() int {
	return t.size
}

// is_matrix returns true if the tensor is a 2D matrix
@[inline]
pub fn (t &CudaTensor[T]) is_matrix() bool {
	return t.rank() == 2
}

// is_square_matrix returns true if the tensor is a square matrix
@[inline]
pub fn (t &CudaTensor[T]) is_square_matrix() bool {
	return t.rank() == 2 && t.shape[0] == t.shape[1]
}

// is_vector returns true if the tensor is a 1D vector
@[inline]
pub fn (t &CudaTensor[T]) is_vector() bool {
	return t.rank() == 1
}

// is_row_major returns true if the tensor uses row-major memory layout
@[inline]
pub fn (t &CudaTensor[T]) is_row_major() bool {
	return t.memory == .row_major
}

// is_col_major returns true if the tensor uses column-major memory layout
@[inline]
pub fn (t &CudaTensor[T]) is_col_major() bool {
	return t.memory == .col_major
}

// is_row_major_contiguous returns true if the tensor is row-major and contiguous
@[inline]
pub fn (t &CudaTensor[T]) is_row_major_contiguous() bool {
	return is_row_major_contiguous(t.shape, t.strides, t.rank())
}

// is_col_major_contiguous returns true if the tensor is column-major and contiguous
@[inline]
pub fn (t &CudaTensor[T]) is_col_major_contiguous() bool {
	return is_col_major_contiguous(t.shape, t.strides, t.rank())
}

// is_contiguous returns true if the tensor is contiguous in either memory layout
@[inline]
pub fn (t &CudaTensor[T]) is_contiguous() bool {
	return t.is_row_major_contiguous() || t.is_col_major_contiguous()
}

// gemm_cuda computes dst = a * b (row-major, no transpose) using cuBLAS.
// Shapes: a is [m, k], b is [k, n], dst is [m, n].
// Requires T = f64 (double precision).
pub fn gemm_cuda(dst &CudaTensor[f64], a &CudaTensor[f64], b &CudaTensor[f64]) ! {
	if a.rank() != 2 || b.rank() != 2 || dst.rank() != 2 {
		return error('gemm_cuda: all tensors must be rank-2 matrices')
	}
	m := a.shape[0]
	k := a.shape[1]
	n := b.shape[1]

	// Convert row-major to column-major for cuBLAS
	a_col := cuda.row_to_col_major(a.data.to_array()!, m, k)
	b_col := cuda.row_to_col_major(b.data.to_array()!, k, n)

	// Get device and run GEMM
	dev := a.data.device
	result_col := compute.gemm_cuda(dev, a_col, b_col, m, n, k)!

	// Convert result back to row-major and copy to destination
	result_row := cuda.col_to_row_major(result_col, m, n)
	unsafe {
		C.cudaMemcpy(dst.data.ptr, result_row.data, int(sizeof(f64)) * m * n,
			C.cuda_memcpy_host_to_device)
	}
}

// gemv_cuda computes y = A * x (matrix-vector product) using cuBLAS.
// Shapes: a is [m, n], x is [n], y is [m].
// Requires T = f64 (double precision).
pub fn gemv_cuda(y &CudaTensor[f64], a &CudaTensor[f64], x &CudaTensor[f64]) ! {
	if a.rank() != 2 || x.rank() != 1 || y.rank() != 1 {
		return error('gemv_cuda: A must be rank-2, x and y must be rank-1')
	}
	m := a.shape[0]
	n := a.shape[1]

	// Convert row-major to column-major for cuBLAS
	a_col := cuda.row_to_col_major(a.data.to_array()!, m, n)

	// Get device and run GEMV
	dev := a.data.device
	result := compute.gemv_cuda(dev, a_col, x.data.to_array()!, m, n)!

	// Copy result to destination
	unsafe {
		C.cudaMemcpy(y.data.ptr, result.data, int(sizeof(f64)) * m,
			C.cuda_memcpy_host_to_device)
	}
}

// relu_cuda applies ReLU activation: max(0, x) on GPU.
// T must be f64.
// Returns a new CudaTensor with the result.
pub fn (t &CudaTensor[f64]) relu_cuda() !&CudaTensor[f64] {
	dev := t.data.device
	input_data := t.data.to_array()!
	result := compute.relu_cuda(dev, input_data)!

	// Create output tensor on GPU
	mut output_storage := &storage.CudaStorage[f64]{
		device: dev
	}
	mut ptr := voidptr(0)
	sz := int(sizeof(f64)) * result.len
	status := C.cudaMalloc(&ptr, sz)
	if status != 0 {
		return error('relu_cuda: cudaMalloc failed with status ${status}')
	}
	output_storage.ptr = ptr
	output_storage.size = sz
	output_storage.count = result.len

	unsafe {
		C.cudaMemcpy(ptr, result.data, sz, C.cuda_memcpy_host_to_device)
	}

	return &CudaTensor[f64]{
		data:    output_storage
		memory:  t.memory
		size:    t.size
		shape:   t.shape
		strides: t.strides
	}
}

// sigmoid_cuda applies sigmoid activation: 1 / (1 + exp(-x)) on GPU.
// T must be f64.
// Returns a new CudaTensor with the result.
pub fn (t &CudaTensor[f64]) sigmoid_cuda() !&CudaTensor[f64] {
	dev := t.data.device
	input_data := t.data.to_array()!
	result := compute.sigmoid_cuda(dev, input_data)!

	// Create output tensor on GPU
	mut output_storage := &storage.CudaStorage[f64]{
		device: dev
	}
	mut ptr := voidptr(0)
	sz := int(sizeof(f64)) * result.len
	status := C.cudaMalloc(&ptr, sz)
	if status != 0 {
		return error('sigmoid_cuda: cudaMalloc failed with status ${status}')
	}
	output_storage.ptr = ptr
	output_storage.size = sz
	output_storage.count = result.len

	unsafe {
		C.cudaMemcpy(ptr, result.data, sz, C.cuda_memcpy_host_to_device)
	}

	return &CudaTensor[f64]{
		data:    output_storage
		memory:  t.memory
		size:    t.size
		shape:   t.shape
		strides: t.strides
	}
}