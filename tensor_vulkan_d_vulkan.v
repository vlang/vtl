module vtl

import vtl.storage
import vsl.vulkan

// VulkanTensor defines a public data structure for this module.

// VulkanTensor defines a public data structure for this module.
@[heap]
pub struct VulkanTensor[T] {
pub mut:
	data    &storage.VulkanStorage[T] = unsafe { nil }
	memory  MemoryFormat
	size    int
	shape   []int
	strides []int
}

// vulkan exposes this operation as part of the public API.
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

// cpu exposes this operation as part of the public API.
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

// vulkan exposes this operation as part of the public API.

// vulkan exposes this operation as part of the public API.
@[inline]
pub fn (t &VulkanTensor[T]) vulkan(params storage.VulkanParams) !&VulkanTensor[T] {
	return t
}

// release exposes this operation as part of the public API.
pub fn (t &VulkanTensor[T]) release() {
	t.data.release()
}

// rank exposes this operation as part of the public API.
pub fn (t &VulkanTensor[T]) rank() int {
	return t.shape.len
}

// numel exposes this operation as part of the public API.
pub fn (t &VulkanTensor[T]) numel() int {
	return t.size
}

// is_matrix exposes this operation as part of the public API.

// is_matrix exposes this operation as part of the public API.
@[inline]
pub fn (t &VulkanTensor[T]) is_matrix() bool {
	return t.rank() == 2
}

// is_square_matrix exposes this operation as part of the public API.

// is_square_matrix exposes this operation as part of the public API.
@[inline]
pub fn (t &VulkanTensor[T]) is_square_matrix() bool {
	return t.rank() == 2 && t.shape[0] == t.shape[1]
}

// is_vector exposes this operation as part of the public API.

// is_vector exposes this operation as part of the public API.
@[inline]
pub fn (t &VulkanTensor[T]) is_vector() bool {
	return t.rank() == 1
}

// is_row_major exposes this operation as part of the public API.

// is_row_major exposes this operation as part of the public API.
@[inline]
pub fn (t &VulkanTensor[T]) is_row_major() bool {
	return t.memory == .row_major
}

// is_col_major exposes this operation as part of the public API.

// is_col_major exposes this operation as part of the public API.
@[inline]
pub fn (t &VulkanTensor[T]) is_col_major() bool {
	return t.memory == .col_major
}

// is_row_major_contiguous exposes this operation as part of the public API.

// is_row_major_contiguous exposes this operation as part of the public API.
@[inline]
pub fn (t &VulkanTensor[T]) is_row_major_contiguous() bool {
	return is_row_major_contiguous(t.shape, t.strides, t.rank())
}

// is_col_major_contiguous exposes this operation as part of the public API.

// is_col_major_contiguous exposes this operation as part of the public API.
@[inline]
pub fn (t &VulkanTensor[T]) is_col_major_contiguous() bool {
	return is_col_major_contiguous(t.shape, t.strides, t.rank())
}

// is_contiguous exposes this operation as part of the public API.

// is_contiguous exposes this operation as part of the public API.
@[inline]
pub fn (t &VulkanTensor[T]) is_contiguous() bool {
	return t.is_row_major_contiguous() || t.is_col_major_contiguous()
}

// relu applies ReLU in-place on the GPU buffer.
pub fn (t &VulkanTensor[f32]) relu() ! {
	dev := t.data.data.device
	vulkan.relu(dev, t.data.data, t.data.data)!
}

// sigmoid applies sigmoid in-place on the GPU buffer.
pub fn (t &VulkanTensor[f32]) sigmoid() ! {
	dev := t.data.data.device
	vulkan.sigmoid(dev, t.data.data, t.data.data)!
}

// gemm_vulkan computes dst = a * b (row-major, no transpose).
// Shapes: a is [m, k], b is [k, n], dst is [m, n].
pub fn gemm_vulkan(dst &VulkanTensor[f32], a &VulkanTensor[f32], b &VulkanTensor[f32]) ! {
	if a.rank() != 2 || b.rank() != 2 || dst.rank() != 2 {
		return error('gemm_vulkan: all tensors must be rank-2 matrices')
	}
	dev := a.data.data.device
	m := u32(a.shape[0])
	k := u32(a.shape[1])
	n := u32(b.shape[1])
	vulkan.gemm(dev, dst.data.data, a.data.data, b.data.data, m, n, k)!
}

// gemv_vulkan computes y = A * x (matrix-vector product).
pub fn gemv_vulkan(y &VulkanTensor[f32], a &VulkanTensor[f32], x &VulkanTensor[f32]) ! {
	if a.rank() != 2 || x.rank() != 1 || y.rank() != 1 {
		return error('gemv_vulkan: a must be rank-2, x and y must be rank-1')
	}
	dev := a.data.data.device
	m := a.shape[0]
	n := a.shape[1]
	vulkan.gemv(dev, y.data.data, a.data.data, x.data.data, m, n)!
}

// vector_add_vulkan computes dst = a + b element-wise.
pub fn vector_add_vulkan(dst &VulkanTensor[f32], a &VulkanTensor[f32], b &VulkanTensor[f32]) ! {
	dev := a.data.data.device
	vulkan.vector_add(dev, dst.data.data, a.data.data, b.data.data)!
}

// vulkan_tensor_zeros_f32 allocates a zero-filled f32 tensor on the given Vulkan device.
pub fn vulkan_tensor_zeros_f32(shape []int, dev &vulkan.Device) !&VulkanTensor[f32] {
	mut t := zeros[f32](shape)
	return t.vulkan(storage.vulkan_params_for_device(dev))!
}

// t transposes a 2-D Vulkan matrix (row-major).
pub fn (t &VulkanTensor[f32]) t() !&VulkanTensor[f32] {
	if t.rank() != 2 {
		return error('VulkanTensor.t: rank must be 2')
	}
	dev := t.data.data.device
	host := t.cpu()!
	th := host.t()!
	return th.vulkan(storage.vulkan_params_for_device(dev))!
}

// gemm computes dst = self * b (row-major GEMM).
pub fn (a &VulkanTensor[f32]) gemm(b &VulkanTensor[f32]) !&VulkanTensor[f32] {
	if a.rank() != 2 || b.rank() != 2 {
		return error('VulkanTensor.gemm: operands must be rank-2')
	}
	m := a.shape[0]
	k := a.shape[1]
	n := b.shape[1]
	if k != b.shape[0] {
		return error('VulkanTensor.gemm: inner dimensions mismatch')
	}
	dev := a.data.data.device
	mut dst := vulkan_tensor_zeros_f32([m, n], dev)!
	gemm_vulkan(dst, a, b)!
	return dst
}
