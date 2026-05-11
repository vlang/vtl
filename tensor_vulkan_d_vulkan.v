module vtl

import storage
import vsl.vulkan
import vsl.compute

@[heap]
pub struct VulkanTensor[T] {
pub mut:
	data    &storage.VulkanStorage[T] = unsafe { nil }
	memory  MemoryFormat
	size    int
	shape   []int
	strides []int
}

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
pub fn (t &VulkanTensor[T]) vulkan() !&VulkanTensor[T] {
	return t
}

pub fn (t &VulkanTensor[T]) release() ! {
	return t.data.release()
}

pub fn (t &VulkanTensor[T]) str() string {
	return 'VulkanTensor shape: ${t.shape} memory: ${t.memory}'
}

pub fn (t &VulkanTensor[T]) rank() int {
	return t.shape.len
}

pub fn (t &VulkanTensor[T]) size() int {
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
		vulkan.gemm(dev, c_buf, a.data.data, b.data.data, u32(m), u32(n), u32(k))!
		return &VulkanTensor[T]{
			data:    &storage.VulkanStorage[T]{ data: c_buf, nelems: m * n }
			memory:  .row_major
			size:    m * n
			shape:   [m, n]
			strides: [n, 1]
		}
	} $else $if T is f64 {
		dev := a.data.data.device
		ctx := compute.new_vulkan_context(dev)
		a_data := a.data.to_array()!
		b_data := b.data.to_array()!
		c_data := compute.gemm_gpu(ctx, a_data, m, k, b_data, b.shape[0], n)!
		c_size := vulkan.DeviceSize(m * n * 8)
		mut c_buf := dev.buffer(c_size)!
		mut c_bytes := []u8{len: int(c_size)}
		unsafe { C.memcpy(c_bytes.data, c_data.data, c_size) }
		c_buf.load(c_bytes)!
		return &VulkanTensor[T]{
			data:    &storage.VulkanStorage[T]{ data: c_buf, nelems: m * n }
			memory:  .row_major
			size:    m * n
			shape:   [m, n]
			strides: [n, 1]
		}
	} $else {
		return error('VulkanTensor.gemm: unsupported type (only f32/f64 supported)')
	}
}

pub fn (t &VulkanTensor[T]) relu() !&VulkanTensor[T] {
	n := t.size
	$if T is f32 {
		dev := t.data.data.device
		size := vulkan.DeviceSize(n * 4)
		mut dst_buf := dev.buffer(size)!
		vulkan.relu(dev, dst_buf, t.data.data)!
		return &VulkanTensor[T]{
			data:    &storage.VulkanStorage[T]{ data: dst_buf, nelems: n }
			memory:  t.memory
			size:    n
			shape:   t.shape.clone()
			strides: t.strides.clone()
		}
	} $else $if T is f64 {
		dev := t.data.data.device
		ctx := compute.new_vulkan_context(dev)
		data := t.data.to_array()!
		result := compute.relu_matrix(ctx, data)!
		size := vulkan.DeviceSize(n * 8)
		mut dst_buf := dev.buffer(size)!
		mut bytes := []u8{len: int(size)}
		unsafe { C.memcpy(bytes.data, result.data, size) }
		dst_buf.load(bytes)!
		return &VulkanTensor[T]{
			data:    &storage.VulkanStorage[T]{ data: dst_buf, nelems: n }
			memory:  t.memory
			size:    n
			shape:   t.shape.clone()
			strides: t.strides.clone()
		}
	} $else {
		return error('VulkanTensor.relu: unsupported type (only f32/f64 supported)')
	}
}

pub fn (t &VulkanTensor[T]) sigmoid() !&VulkanTensor[T] {
	n := t.size
	$if T is f32 {
		dev := t.data.data.device
		size := vulkan.DeviceSize(n * 4)
		mut dst_buf := dev.buffer(size)!
		vulkan.sigmoid(dev, dst_buf, t.data.data)!
		return &VulkanTensor[T]{
			data:    &storage.VulkanStorage[T]{ data: dst_buf, nelems: n }
			memory:  t.memory
			size:    n
			shape:   t.shape.clone()
			strides: t.strides.clone()
		}
	} $else $if T is f64 {
		dev := t.data.data.device
		ctx := compute.new_vulkan_context(dev)
		data := t.data.to_array()!
		result := compute.sigmoid_matrix(ctx, data)!
		size := vulkan.DeviceSize(n * 8)
		mut dst_buf := dev.buffer(size)!
		mut bytes := []u8{len: int(size)}
		unsafe { C.memcpy(bytes.data, result.data, size) }
		dst_buf.load(bytes)!
		return &VulkanTensor[T]{
			data:    &storage.VulkanStorage[T]{ data: dst_buf, nelems: n }
			memory:  t.memory
			size:    n
			shape:   t.shape.clone()
			strides: t.strides.clone()
		}
	} $else {
		return error('VulkanTensor.sigmoid: unsupported type (only f32/f64 supported)')
	}
}

pub fn (a &VulkanTensor[T]) gemv(x &VulkanTensor[T]) !&VulkanTensor[T] {
	if a.rank() != 2 {
		return error('VulkanTensor.gemv: a must be rank-2')
	}
	m := a.shape[0]
	n := a.shape[1]
	nx := if x.rank() == 1 { x.shape[0] } else { x.shape[0] }
	if n != nx {
		return error('VulkanTensor.gemv: inner dimension mismatch: a(${m}x${n}) * x(${nx})')
	}
	$if T is f32 {
		dev := a.data.data.device
		y_size := vulkan.DeviceSize(m * 4)
		mut y_buf := dev.buffer(y_size)!
		vulkan.gemv(dev, y_buf, a.data.data, x.data.data, m, n)!
		return &VulkanTensor[T]{
			data:    &storage.VulkanStorage[T]{ data: y_buf, nelems: m }
			memory:  .row_major
			size:    m
			shape:   [m]
			strides: [1]
		}
	} $else $if T is f64 {
		dev := a.data.data.device
		ctx := compute.new_vulkan_context(dev)
		a_data := a.data.to_array()!
		x_data := x.data.to_array()!
		result := compute.gemv_gpu(ctx, a_data, m, n, x_data)!
		y_size := vulkan.DeviceSize(m * 8)
		mut y_buf := dev.buffer(y_size)!
		mut bytes := []u8{len: int(y_size)}
		unsafe { C.memcpy(bytes.data, result.data, int(y_size)) }
		y_buf.load(bytes)!
		return &VulkanTensor[T]{
			data:    &storage.VulkanStorage[T]{ data: y_buf, nelems: m }
			memory:  .row_major
			size:    m
			shape:   [m]
			strides: [1]
		}
	} $else {
		return error('VulkanTensor.gemv: unsupported type (only f32/f64 supported)')
	}
}

pub fn vector_add_vulkan[T](dst &VulkanTensor[T], a &VulkanTensor[T], b &VulkanTensor[T]) !&VulkanTensor[T] {
	if a.size != b.size || a.size != dst.size {
		return error('VulkanTensor.vector_add: size mismatch: dst(${dst.size}), a(${a.size}), b(${b.size})')
	}
	$if T is f32 {
		dev := dst.data.data.device
		size := vulkan.DeviceSize(dst.size * 4)
		mut out_buf := dev.buffer(size)!
		vulkan.vector_add(dev, out_buf, a.data.data, b.data.data)!
		return &VulkanTensor[T]{
			data:    &storage.VulkanStorage[T]{ data: out_buf, nelems: dst.size }
			memory:  dst.memory
			size:    dst.size
			shape:   dst.shape.clone()
			strides: dst.strides.clone()
		}
	} $else $if T is f64 {
		a_data := a.data.to_array()!
		b_data := b.data.to_array()!
		mut result := []f64{len: dst.size}
		for i in 0 .. dst.size {
			result[i] = a_data[i] + b_data[i]
		}
		dev := dst.data.data.device
		size := vulkan.DeviceSize(dst.size * 8)
		mut out_buf := dev.buffer(size)!
		mut bytes := []u8{len: int(size)}
		unsafe { C.memcpy(bytes.data, result.data, int(size)) }
		out_buf.load(bytes)!
		return &VulkanTensor[T]{
			data:    &storage.VulkanStorage[T]{ data: out_buf, nelems: dst.size }
			memory:  dst.memory
			size:    dst.size
			shape:   dst.shape.clone()
			strides: dst.strides.clone()
		}
	} $else {
		return error('VulkanTensor.vector_add: unsupported type (only f32/f64 supported)')
	}
}
