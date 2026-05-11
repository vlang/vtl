module autograd

// GPU-accelerated MatMul backward gate.
// dA = grad_out @ B^T via d_gemm_da kernel
// dB = A^T @ grad_out via d_gemm_db kernel

import vtl
import vtl.la
import storage
import vsl.vulkan

pub struct MatMulGateVulkan[T] {
pub:
	a      &Variable[T]              = unsafe { nil }
	b      &Variable[T]              = unsafe { nil }
	params storage.VulkanStorageParams
}

pub fn matmul_gate_vulkan[T](a &Variable[T], b &Variable[T], params storage.VulkanStorageParams) &MatMulGateVulkan[T] {
	return &MatMulGateVulkan[T]{a: a, b: b, params: params}
}

pub fn (g &MatMulGateVulkan[T]) backward[T](payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	a_val := g.a.value
	b_val := g.b.value
	if a_val.rank() != 2 || b_val.rank() != 2 {
		return error('MatMulGateVulkan: inputs must be 2D')
	}
	m := u32(a_val.shape[0])
	k := u32(a_val.shape[1])
	n := u32(b_val.shape[1])
	$if T is f32 {
		dev := g.params.device
		// Allocate output buffers
		size_da := vulkan.DeviceSize(u64(m) * u64(k) * 4) // dA: M x K
		size_db := vulkan.DeviceSize(u64(k) * u64(n) * 4) // dB: K x N
		vk_grad := gradient.vulkan(g.params)!
		defer { vk_grad.release() or {} }
		vk_a := a_val.vulkan(g.params)!
		defer { vk_a.release() or {} }
		vk_b := b_val.vulkan(g.params)!
		defer { vk_b.release() or {} }
		mut da_buf := dev.buffer(size_da)!
		defer { da_buf.release() }
		mut db_buf := dev.buffer(size_db)!
		defer { db_buf.release() }
		// dA = grad_out @ B^T
		vulkan.d_gemm_da(dev, da_buf, vk_grad.data.data, vk_b.data.data, m, n, k)!
		// dB = A^T @ grad_out
		vulkan.d_gemm_db(dev, db_buf, vk_grad.data.data, vk_a.data.data, m, n, k)!
		// Read back dA
		mut raw_da := []u8{len: int(size_da)}
		raw_da = da_buf.store(mut raw_da)!
		mut vals_da := []T{len: int(u64(m) * u64(k))}
		for i in 0 .. vals_da.len {
			unsafe { vals_da[i] = T(*(&f32(&raw_da[i * 4]))) }
		}
		r0 := vtl.from_array[T](vals_da, [int(m), int(k)], vtl.TensorData{ memory: .row_major })!
		// Read back dB
		mut raw_db := []u8{len: int(size_db)}
		raw_db = db_buf.store(mut raw_db)!
		mut vals_db := []T{len: int(u64(k) * u64(n))}
		for i in 0 .. vals_db.len {
			unsafe { vals_db[i] = T(*(&f32(&raw_db[i * 4]))) }
		}
		r1 := vtl.from_array[T](vals_db, [int(k), int(n)], vtl.TensorData{ memory: .row_major })!
		return [r0, r1]
	} $else {
		// CPU fallback
		r0 := la.matmul[T](gradient, b_val.t()!)!
		r1 := la.matmul[T](a_val.t()!, gradient)!
		return [r0, r1]
	}
}

pub fn (g &MatMulGateVulkan[T]) cache[T](mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	b := args[1]
	match a {
		Variable[T] {
			match b {
				Variable[T] {
					result.grad = vtl.zeros_like[T](result.value)
					result.requires_grad = true
					register[T]('MatMulVulkan', g, result, [a, b])!
				}
				else {
					return error('MatMulGateVulkan: b must be a Variable')
				}
			}
		}
		else {
			return error('MatMulGateVulkan: a must be a Variable')
		}
	}
}
