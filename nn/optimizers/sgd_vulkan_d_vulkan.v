module optimizers

// GPU-accelerated SGD optimizer step using VSL vector_add and scale kernels.
// Compiles only with -d vulkan.
import vtl
import vtl.autograd
import storage
import vsl.vulkan

// sgd_step_vulkan performs param = param - lr * grad on GPU for f32 tensors.
// Falls back to CPU for other types.
pub fn sgd_step_vulkan[T](mut v autograd.Variable[T], learning_rate f64, params storage.VulkanStorageParams) ! {
	if !v.requires_grad {
		return
	}
	$if T is f32 {
		dev := params.device
		n := v.value.size
		size := vulkan.DeviceSize(u64(n) * 4)
		// Upload param and grad to GPU
		vk_param := v.value.vulkan(params)!
		defer { vk_param.release() or {} }
		vk_grad := v.grad.vulkan(params)!
		defer { vk_grad.release() or {} }
		// scaled_grad = lr * grad
		mut scaled_buf := dev.buffer(size)!
		defer { scaled_buf.release() }
		vulkan.scale(dev, scaled_buf, vk_grad.data.data, f32(learning_rate))!
		// new_param = param - scaled_grad  →  vector_add(param, -scaled_grad)
		// negate scaled_grad first: scale by -1
		mut neg_scaled_buf := dev.buffer(size)!
		defer { neg_scaled_buf.release() }
		vulkan.scale(dev, neg_scaled_buf, scaled_buf, f32(-1.0))!
		// result = param + neg_scaled
		mut result_buf := dev.buffer(size)!
		defer { result_buf.release() }
		vulkan.vector_add(dev, result_buf, vk_param.data.data, neg_scaled_buf)!
		// Read back and update CPU tensor
		mut raw := []u8{len: n * 4}
		raw = result_buf.store(mut raw)!
		for i in 0 .. n {
			unsafe { v.value.data.set(i, T(*(&f32(&raw[i * 4])))) }
		}
		v.grad = vtl.zeros_like[T](v.value)
	} $else {
		// CPU fallback
		mut iters, _ := v.value.iterators([v.grad])!
		for {
			vals, i := iters.next() or { break }
			val := vals[0] - vtl.cast[T](learning_rate) * vals[1]
			v.value.set(i, val)
		}
		v.grad = vtl.zeros_like[T](v.value)
	}
}
