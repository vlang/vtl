module layers

import storage
import vtl
import vsl.vulkan

// softmax_forward_vulkan computes numerically-stable softmax over the flat
// element space of the input tensor on the Vulkan GPU (f32 native).
// f64 falls back to a CPU pass via byte-level reinterpretation.
pub fn softmax_forward_vulkan[T](x &vtl.Tensor[T], params storage.VulkanStorageParams) !&vtl.Tensor[T] {
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
		vulkan.softmax(dev, out_buf, vk.data.data, u32(n))!
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
		return error('softmax_forward_vulkan: only f32 is GPU-accelerated (got ${T.name})')
	}
}

// layernorm_forward_vulkan computes layer normalisation on GPU (f32 native).
// Returns normalised output only; caller applies gamma/beta on CPU.
pub fn layernorm_forward_vulkan[T](x &vtl.Tensor[T], eps f32, params storage.VulkanStorageParams) !&vtl.Tensor[T] {
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
		vulkan.layernorm(dev, out_buf, vk.data.data, u32(n), eps)!
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
		return error('layernorm_forward_vulkan: only f32 is GPU-accelerated (got ${T.name})')
	}
}

// reduce_sum_vulkan returns partial per-workgroup sums of x on GPU (f32 native).
// For a global sum, add the returned partial sums on CPU.
pub fn reduce_sum_vulkan[T](x &vtl.Tensor[T], params storage.VulkanStorageParams) ![]T {
	n := x.size()
	wg_size := u32(256)
	num_groups := int((u32(n) + wg_size - 1) / wg_size)
	$if T is f32 {
		vk := x.vulkan(params)!
		defer {
			vk.release() or {}
		}
		dev := vk.data.data.device
		mut out_buf := dev.buffer(vulkan.DeviceSize(u64(num_groups) * 4))!
		defer {
			out_buf.release()
		}
		vulkan.reduce(dev, out_buf, vk.data.data, u32(n), .sum)!
		mut raw := []u8{len: num_groups * 4}
		raw = out_buf.store(mut raw)!
		mut partial := []T{len: num_groups}
		for i in 0 .. num_groups {
			unsafe {
				partial[i] = T(*(&f32(&raw[i * 4])))
			}
		}
		return partial
	} $else {
		return error('reduce_sum_vulkan: only f32 is GPU-accelerated (got ${T.name})')
	}
}
