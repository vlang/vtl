module autograd

// GPU-accelerated backward gates for reduction operations (Sum, Mean).
// Uses broadcast_grad VSL kernel to expand gradient from reduced shape
// back to the original input shape.
import vtl
import storage
import vsl.vulkan

// SumGateVulkan: backward expands scalar/reduced gradient to input shape.
// Uses broadcast_grad kernel: grad_in[i] = grad_out[i % n] * 1.0
pub struct SumGateVulkan[T] {
pub:
	shape  []int
	axis   int
	params storage.VulkanStorageParams
}

pub fn sum_gate_vulkan[T](shape []int, axis int, params storage.VulkanStorageParams) &SumGateVulkan[T] {
	return &SumGateVulkan[T]{
		shape:  shape
		axis:   axis
		params: params
	}
}

pub fn (g &SumGateVulkan[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	$if T is f32 {
		dev := g.params.device
		// Total elements in the original input
		total := g.shape.reduce(fn (a int, b int) int {
			return a * b
		}, 1)
		// Number of elements in the gradient (reduced dimension)
		n_grad := gradient.size
		size_in := vulkan.DeviceSize(u64(total) * 4)
		size_out := vulkan.DeviceSize(u64(n_grad) * 4)
		vk_grad := gradient.vulkan(g.params)!
		defer { vk_grad.release() or {} }
		mut in_buf := dev.buffer(size_in)!
		defer { in_buf.release() }
		vulkan.broadcast_grad(dev, in_buf, vk_grad.data.data, u32(n_grad), f32(1.0))!
		mut raw := []u8{len: total * 4}
		raw = in_buf.store(mut raw)!
		mut vals := []T{len: total}
		for i in 0 .. total {
			unsafe {
				vals[i] = T(*(&f32(&raw[i * 4])))
			}
		}
		r0 := vtl.from_array[T](vals, g.shape, vtl.TensorData{ memory: .row_major })!
		return [r0]
	} $else {
		r0 := gradient.broadcast_to[T](g.shape)!
		return [r0]
	}
}

pub fn (g &SumGateVulkan[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('SumVulkan', g, result, [a])!
		}
		else {
			return error('SumGateVulkan: a must be a Variable')
		}
	}
}

// MeanGateVulkan: backward expands and scales by 1/num_elems.
pub struct MeanGateVulkan[T] {
pub:
	shape     []int
	axis      int
	num_elems int
	params    storage.VulkanStorageParams
}

pub fn mean_gate_vulkan[T](shape []int, axis int, num_elems int, params storage.VulkanStorageParams) &MeanGateVulkan[T] {
	return &MeanGateVulkan[T]{
		shape:     shape
		axis:      axis
		num_elems: num_elems
		params:    params
	}
}

pub fn (g &MeanGateVulkan[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	$if T is f32 {
		dev := g.params.device
		total := g.shape.reduce(fn (a int, b int) int {
			return a * b
		}, 1)
		n_grad := gradient.size
		size_in := vulkan.DeviceSize(u64(total) * 4)
		vk_grad := gradient.vulkan(g.params)!
		defer { vk_grad.release() or {} }
		mut in_buf := dev.buffer(size_in)!
		defer { in_buf.release() }
		scale := f32(1.0) / f32(g.num_elems)
		vulkan.broadcast_grad(dev, in_buf, vk_grad.data.data, u32(n_grad), scale)!
		mut raw := []u8{len: total * 4}
		raw = in_buf.store(mut raw)!
		mut vals := []T{len: total}
		for i in 0 .. total {
			unsafe {
				vals[i] = T(*(&f32(&raw[i * 4])))
			}
		}
		r0 := vtl.from_array[T](vals, g.shape, vtl.TensorData{ memory: .row_major })!
		return [r0]
	} $else {
		broadcasted := gradient.broadcast_to[T](g.shape)!
		scale := vtl.cast[T](g.num_elems)
		r0 := broadcasted.divide_scalar[T](scale)!
		return [r0]
	}
}

pub fn (g &MeanGateVulkan[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			register[T]('MeanVulkan', g, result, [a])!
		}
		else {
			return error('MeanGateVulkan: a must be a Variable')
		}
	}
}
