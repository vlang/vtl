module layers

import storage
import vtl
import vsl.vulkan

// gelu_forward_vulkan computes GELU activation on GPU (f32 native).
// Uses tanh approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// f64 falls back to CPU.
pub fn gelu_forward_vulkan[T](x &vtl.Tensor[T], params storage.VulkanStorageParams) !&vtl.Tensor[T] {
	n := x.size()
	$if T is f32 {
		vk := x.vulkan(params)!
		defer {
			vk.release() or {}
		}
		dev := vk.data.data.device
		mut out_buf := dev.buffer(vulkan.DeviceSize(u64(n) * 4))!
		defer {
			out_buf.release()
		}
		vulkan.gelu(dev, out_buf, vk.data.data, u32(n))!
		mut raw := []u8{len: n * 4}
		raw = out_buf.store(mut raw)!
		mut vals := []T{len: n}
		for i in 0 .. n {
			unsafe {
				vals[i] = T(*(&f32(&raw[i * 4])))
			}
		}
		return vtl.from_array[T](vals, x.shape, vtl.TensorData{ memory: .row_major })!
	} $else {
		return error('gelu_forward_vulkan: only f32 is GPU-accelerated (got ${T.name})')
	}
}
