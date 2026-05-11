module layers

import vtl
import vsl.vulkan

pub struct BatchNorm1DLayerVulkan[T] {
	eps    f32
	device &vulkan.Device = unsafe { nil }
}

// batchnorm1d_forward_vulkan normalises input[N, C] along the batch (N) dimension
// per feature on the Vulkan GPU (f32 native). Returns the normalised tensor only;
// caller applies gamma/beta on CPU. f64 returns an error.
pub fn batchnorm1d_forward_vulkan[T](input &vtl.Tensor[T], eps f32, dev &vulkan.Device) !&vtl.Tensor[T] {
	if input.shape.len != 2 {
		return error('batchnorm1d_forward_vulkan: expected 2-D input [N, C], got shape ${input.shape}')
	}
	n := u32(input.shape[0])
	c := u32(input.shape[1])
	total := int(n * c)

	$if T !is f32 {
		return error('batchnorm1d_forward_vulkan: only f32 is GPU-accelerated (got ${T.name})')
	}

	mut src_bytes := []u8{len: total * 4}
	for idx in 0 .. total {
		row := idx / int(c)
		col := idx % int(c)
		val := f32(input.get([row, col]))
		unsafe {
			*(&f32(&src_bytes[idx * 4])) = val
		}
	}

	mut src_buf := dev.buffer(vulkan.DeviceSize(u64(total) * 4))!
	defer { src_buf.release() }
	src_buf.load(src_bytes)!

	mut dst_buf := dev.buffer(vulkan.DeviceSize(u64(total) * 4))!
	defer { dst_buf.release() }

	vulkan.batchnorm1d(dev, dst_buf, src_buf, n, c, eps)!

	mut raw := []u8{len: total * 4}
	dst_buf.store(mut raw)!

	mut vals := []T{len: total}
	for idx in 0 .. total {
		unsafe {
			vals[idx] = T(*(&f32(&raw[idx * 4])))
		}
	}

	out := vtl.from_1d[T](vals, vtl.TensorData{})!
	return out.reshape([int(n), int(c)])!
}

pub fn (layer &BatchNorm1DLayerVulkan[T]) forward(input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	if layer.device == unsafe { nil } {
		return error('BatchNorm1DLayerVulkan: device is nil')
	}
	return batchnorm1d_forward_vulkan[T](input, layer.eps, layer.device)!
}
