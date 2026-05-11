module vtl

import storage
import vsl.vulkan
import vsl.compute

// VulkanTensor is the GPU tensor structure, analogous to VclTensor.
// Mirrors the pattern from tensor_vcl_d_vcl.v.
@[heap]
pub struct VulkanTensor[T] {
pub mut:
	data    &storage.VulkanStorage[T] = unsafe { nil }
	memory  MemoryFormat
	size    int
	shape   []int
	strides []int
}

// vulkan returns a VulkanTensor from a CPU Tensor.
// The tensor is copied to row-major before upload.
pub fn (t &Tensor[T]) vulkan(params storage.VulkanStorageParams) !&VulkanTensor[T] {
	row_tensor := t.copy(.row_major)
	vk_data := row_tensor.data.vulkan(params)!
	return &VulkanTensor[T]{
		data:    vk_data
		memory:  row_tensor.memory
		size:    row_tensor.size
		shape:   row_tensor.shape
		strides: row_tensor.strides
	}
}

// cpu returns a CPU Tensor from a VulkanTensor.
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

// vulkan returns self (for interface compatibility).
@[inline]
pub fn (t &VulkanTensor[T]) vulkan() !&VulkanTensor[T] {
	return t
}

// release frees the GPU data.
pub fn (t &VulkanTensor[T]) release() ! {
	return t.data.release()
}

// str returns a string representation of the tensor metadata.
pub fn (t &VulkanTensor[T]) str() string {
	return 'VulkanTensor shape: ${t.shape} memory: ${t.memory}'
}

// rank returns the number of dimensions.
pub fn (t &VulkanTensor[T]) rank() int {
	return t.shape.len
}

// size returns the number of allocated elements.
pub fn (t &VulkanTensor[T]) size() int {
	return t.size
}

// is_matrix returns if the tensor is a 2D matrix.
@[inline]
pub fn (t &VulkanTensor[T]) is_matrix() bool {
	return t.rank() == 2
}

// is_square_matrix returns if the tensor is a square matrix.
@[inline]
pub fn (t &VulkanTensor[T]) is_square_matrix() bool {
	return t.rank() == 2 && t.shape[0] == t.shape[1]
}

// is_vector returns if the tensor is a 1D vector.
@[inline]
pub fn (t &VulkanTensor[T]) is_vector() bool {
	return t.rank() == 1
}

// is_row_major returns if the tensor is row-major.
@[inline]
pub fn (t &VulkanTensor[T]) is_row_major() bool {
	return t.memory == .row_major
}

// is_col_major returns if the tensor is column-major.
@[inline]
pub fn (t &VulkanTensor[T]) is_col_major() bool {
	return t.memory == .col_major
}

// is_row_major_contiguous checks if data is contiguous in row-major order.
@[inline]
pub fn (t &VulkanTensor[T]) is_row_major_contiguous() bool {
	return is_row_major_contiguous(t.shape, t.strides, t.rank())
}

// is_col_major_contiguous checks if data is contiguous in column-major order.
@[inline]
pub fn (t &VulkanTensor[T]) is_col_major_contiguous() bool {
	return is_col_major_contiguous(t.shape, t.strides, t.rank())
}

// is_contiguous checks if data is contiguous in any layout.
@[inline]
pub fn (t &VulkanTensor[T]) is_contiguous() bool {
	return t.is_row_major_contiguous() || t.is_col_major_contiguous()
}

// --------------------------------------------------------------------------
// GPU Compute Operations
// --------------------------------------------------------------------------

// gemm performs GPU-accelerated matrix multiply: C = A * B
// A: [M x K], B: [K x N], returns C: [M x N] (all row-major).
// For f32: direct GPU dispatch (no CPU round-trip).
// For f64: uses VSL compute layer (f64→f32→GPU→f32→f64 bridge).
pub fn (a &VulkanTensor[T]) gemm(b &VulkanTensor[T]) !&VulkanTensor[T] {
	if a.rank() != 2 || b.rank() != 2 {
		return error('VulkanTensor.gemm: both tensors must be 2D, got A(${a.rank()}D), B(${b.rank()}D)')
	}
	m := a.shape[0]
	k := a.shape[1]
	n := b.shape[1]
	if k != b.shape[0] {
		return error('VulkanTensor.gemm: dimension mismatch: A(${m}x${k}) * B(${b.shape[0]}x${n})')
	}
	$if T is f32 {
		dev := a.data.data.device
		c_size := vulkan.DeviceSize(m * n * 4)
		mut c_buf := dev.buffer(c_size)!
		vulkan.gemm(c_buf, a.data.data, b.data.data, u32(m), u32(n), u32(k))!
		return &VulkanTensor[T]{
			data: &storage.VulkanStorage[T]{ data: c_buf, nelems: m * n }
			memory: .row_major
			size: m * n
			shape: [m, n]
			strides: [n, 1]
		}
	} $else $if T is f64 {
		a_data := a.data.to_array()!
		b_data := b.data.to_array()!
		c_data := compute.gemm_gpu(a_data, m, k, b_data, b.shape[0], n)!
		dev := a.data.data.device
		c_size := vulkan.DeviceSize(m * n * 8)
		mut c_buf := dev.buffer(c_size)!
		mut c_bytes := []u8{len: int(c_size)}
		unsafe { C.memcpy(c_bytes.data, c_data.data, c_size) }
		c_buf.load(c_bytes)!
		return &VulkanTensor[T]{
			data: &storage.VulkanStorage[T]{ data: c_buf, nelems: m * n }
			memory: .row_major
			size: m * n
			shape: [m, n]
			strides: [n, 1]
		}
	} $else {
		return error('VulkanTensor.gemm: unsupported type (only f32/f64 supported)')
	}
}

// relu applies ReLU activation on GPU: dst[i] = max(0, src[i]).
pub fn (t &VulkanTensor[T]) relu() !&VulkanTensor[T] {
	n := t.size
	$if T is f32 {
		dev := t.data.data.device
		size := vulkan.DeviceSize(n * 4)
		mut dst_buf := dev.buffer(size)!
		vulkan.relu(dst_buf, t.data.data)!
		return &VulkanTensor[T]{
			data: &storage.VulkanStorage[T]{ data: dst_buf, nelems: n }
			memory: t.memory
			size: n
			shape: t.shape.clone()
			strides: t.strides.clone()
		}
	} $else $if T is f64 {
		data := t.data.to_array()!
		result := compute.relu_matrix(data)!
		dev := t.data.data.device
		size := vulkan.DeviceSize(n * 8)
		mut dst_buf := dev.buffer(size)!
		mut bytes := []u8{len: int(size)}
		unsafe { C.memcpy(bytes.data, result.data, size) }
		dst_buf.load(bytes)!
		return &VulkanTensor[T]{
			data: &storage.VulkanStorage[T]{ data: dst_buf, nelems: n }
			memory: t.memory
			size: n
			shape: t.shape.clone()
			strides: t.strides.clone()
		}
	} $else {
		return error('VulkanTensor.relu: unsupported type (only f32/f64 supported)')
	}
}

// sigmoid applies sigmoid activation on GPU: dst[i] = 1/(1+exp(-src[i])).
pub fn (t &VulkanTensor[T]) sigmoid() !&VulkanTensor[T] {
	n := t.size
	$if T is f32 {
		dev := t.data.data.device
		size := vulkan.DeviceSize(n * 4)
		mut dst_buf := dev.buffer(size)!
		vulkan.sigmoid(dst_buf, t.data.data)!
		return &VulkanTensor[T]{
			data: &storage.VulkanStorage[T]{ data: dst_buf, nelems: n }
			memory: t.memory
			size: n
			shape: t.shape.clone()
			strides: t.strides.clone()
		}
	} $else $if T is f64 {
		data := t.data.to_array()!
		result := compute.sigmoid_matrix(data)!
		dev := t.data.data.device
		size := vulkan.DeviceSize(n * 8)
		mut dst_buf := dev.buffer(size)!
		mut bytes := []u8{len: int(size)}
		unsafe { C.memcpy(bytes.data, result.data, size) }
		dst_buf.load(bytes)!
		return &VulkanTensor[T]{
			data: &storage.VulkanStorage[T]{ data: dst_buf, nelems: n }
			memory: t.memory
			size: n
			shape: t.shape.clone()
			strides: t.strides.clone()
		}
	} $else {
		return error('VulkanTensor.sigmoid: unsupported type (only f32/f64 supported)')
	}
}
