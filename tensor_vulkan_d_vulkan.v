module vtl

import storage
import vsl.vulkan

@[heap]
pub struct VulkanTensor[T] {
pub mut:
	data    &storage.VulkanStorage[T] = unsafe { nil }
	memory  MemoryFormat
	size    int
	shape   []int
	strides []int
}

pub fn (t &Tensor[T]) vulkan(params storage.VulkanParams) !&VulkanTensor[T] {
	row_tensor := t.copy(.row_major)
	vkdata := row_tensor.data.vulkan(params)!
	return &VulkanTensor[T]{
		data:    vkdata
		memory:  row_tensor.memory
		size:    row_tensor.size
		shape:   row_tensor.shape
		strides: row_tensor.strides
	}
}

pub fn (t &VulkanTensor[T]) cpu() !&Tensor[T] {
	data := t.data.cpu()!
	return &Tensor[T]{
		data:    data
		memory:  t.memory
		size:    t.size
		shape:   t.shape
		strides: t.strides
	}
}

@[inline]
pub fn (t &VulkanTensor[T]) vulkan(params storage.VulkanParams) !&VulkanTensor[T] {
	return t
}

pub fn (t &VulkanTensor[T]) release() {
	t.data.release()
}

pub fn (t &VulkanTensor[T]) rank() int {
	return t.shape.len
}

pub fn (t &VulkanTensor[T]) numel() int {
	return t.size
}

@[inline]
pub fn (t &VulkanTensor[T]) is_matrix() bool {
	return t.rank() == 2
}

@[inline]
pub fn (t &VulkanTensor[T]) is_square_matrix() bool {
	return t.rank() == 2 && t.shape[0] == t.shape[1]
}

@[inline]
pub fn (t &VulkanTensor[T]) is_vector() bool {
	return t.rank() == 1
}

@[inline]
pub fn (t &VulkanTensor[T]) is_row_major() bool {
	return t.memory == .row_major
}

@[inline]
pub fn (t &VulkanTensor[T]) is_col_major() bool {
	return t.memory == .col_major
}

@[inline]
pub fn (t &VulkanTensor[T]) is_row_major_contiguous() bool {
	return is_row_major_contiguous(t.shape, t.strides, t.rank())
}

@[inline]
pub fn (t &VulkanTensor[T]) is_col_major_contiguous() bool {
	return is_col_major_contiguous(t.shape, t.strides, t.rank())
}

@[inline]
pub fn (t &VulkanTensor[T]) is_contiguous() bool {
	return t.is_row_major_contiguous() || t.is_col_major_contiguous()
}

// relu_vulkan applies ReLU in-place on the GPU buffer.
// T must be f32.
pub fn (t &VulkanTensor[f32]) relu() ! {
	vulkan.relu(t.data.data, t.data.data)!
}

// sigmoid_vulkan applies sigmoid in-place on the GPU buffer.
// T must be f32.
pub fn (t &VulkanTensor[f32]) sigmoid() ! {
	vulkan.sigmoid(t.data.data, t.data.data)!
}

// gemm_vulkan computes dst = a * b (row-major, no transpose).
// Shapes: a is [m, k], b is [k, n], dst is [m, n].
pub fn gemm_vulkan(dst &VulkanTensor[f32], a &VulkanTensor[f32], b &VulkanTensor[f32]) ! {
	if a.rank() != 2 || b.rank() != 2 || dst.rank() != 2 {
		return error('gemm_vulkan: all tensors must be rank-2 matrices')
	}
	m := u32(a.shape[0])
	k := u32(a.shape[1])
	n := u32(b.shape[1])
	vulkan.gemm(dst.data.data, a.data.data, b.data.data, m, n, k)!
}

// gemv_vulkan computes y = A * x (matrix-vector product).
// Shapes: a is [m, n], x is [n], y is [m].
pub fn gemv_vulkan(y &VulkanTensor[f32], a &VulkanTensor[f32], x &VulkanTensor[f32]) ! {
	if a.rank() != 2 || x.rank() != 1 || y.rank() != 1 {
		return error('gemv_vulkan: a must be rank-2, x and y must be rank-1')
	}
	m := a.shape[0]
	n := a.shape[1]
	vulkan.gemv(y.data.data, a.data.data, x.data.data, m, n)!
}

// vector_add_vulkan computes dst = a + b element-wise.
pub fn vector_add_vulkan(dst &VulkanTensor[f32], a &VulkanTensor[f32], b &VulkanTensor[f32]) ! {
	vulkan.vector_add(dst.data.data, a.data.data, b.data.data)!
}
