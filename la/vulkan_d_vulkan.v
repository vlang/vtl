module la

import vtl
import vsl.vulkan

// matmul_vulkan performs matrix multiplication on Vulkan GPU.
// A: [m, k], B: [k, n] → C: [m, n]
// Accepts any tensor; converts to f32 internally.
pub fn matmul_vulkan[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[T] {
	a.assert_matrix()!
	b.assert_matrix()!
	if a.shape[1] != b.shape[0] {
		return error('Invalid shapes for matrix multiplication ${a.shape} and ${b.shape}')
	}

	m := u32(a.shape[0])
	k := u32(a.shape[1])
	n := u32(b.shape[1])

	// Create Vulkan device
	mut dev := vulkan.new_device() or { return error('matmul_vulkan: no Vulkan device available') }
	defer { dev.release() }

	// Allocate GPU buffers
	mut a_buf := dev.buffer(vulkan.DeviceSize(a.size() * 4))!
	defer { a_buf.release() }
	mut b_buf := dev.buffer(vulkan.DeviceSize(b.size() * 4))!
	defer { b_buf.release() }
	mut c_buf := dev.buffer(vulkan.DeviceSize(m * n * 4))!
	defer { c_buf.release() }

	// Convert A and B to f32 and upload
	a_f32 := a.to_f32()
	b_f32 := b.to_f32()

	mut a_bytes := []u8{len: int(a.size() * 4)}
	mut b_bytes := []u8{len: int(b.size() * 4)}

	for i in 0 .. a_f32.len {
		val := a_f32[i]
		unsafe {
			*(&f32(&a_bytes[i * 4])) = val
		}
	}
	for i in 0 .. b_f32.len {
		val := b_f32[i]
		unsafe {
			*(&f32(&b_bytes[i * 4])) = val
		}
	}

	a_buf.load(a_bytes)!
	b_buf.load(b_bytes)!

	// GEMM on GPU: C = A * B
	vulkan.gemm(dev, c_buf, a_buf, b_buf, m, n, k)!

	// Download result
	mut c_bytes := []u8{len: int(m * n * 4)}
	c_buf.store(mut c_bytes)!

	// Convert bytes back to f32 then to T
	mut c_f32 := []f32{len: int(m * n)}
	for i in 0 .. int(m * n) {
		unsafe {
			c_f32[i] = *(&f32(&c_bytes[i * 4]))
		}
	}

	// Convert f32 to T
	c_data := c_f32.map(vtl.cast[T](it))

	// Return as tensor with correct shape
	mut c_tensor := vtl.from_1d[T](c_data, vtl.TensorData{}) or { return err }
	return c_tensor.reshape([int(m), int(n)])!
}
