module la

import vtl
import vsl.vulkan
import vtl.storage

// matmul_vulkan performs matrix multiplication on Vulkan GPU.
// A: [m, k], B: [k, n] → C: [m, n]
// Both inputs must use VulkanStorage.
pub fn matmul_vulkan[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[T] {
	a.assert_matrix()!
	b.assert_matrix()!
	if a.shape[1] != b.shape[0] {
		return error('Invalid shapes for matrix multiplication ${a.shape} and ${b.shape}')
	}

	m := u32(a.shape[0])
	k := u32(a.shape[1])
	n := u32(b.shape[1])

	// Get Vulkan device from input storage
	a_vk := match a.storage {
		storage.VulkanStorage[T] { a.storage as storage.VulkanStorage[T] }
		else { return error('matmul_vulkan: a must use VulkanStorage') }
	}
	dev := a_vk.params.device

	// Allocate GPU buffers
	mut a_buf := dev.buffer(vulkan.DeviceSize(a.size() * 4))!
	defer { a_buf.release() }
	mut b_buf := dev.buffer(vulkan.DeviceSize(b.size() * 4))!
	defer { b_buf.release() }
	mut c_buf := dev.buffer(vulkan.DeviceSize(m * n * 4))!
	defer { c_buf.release() }

	// Upload A and B to GPU (convert to f32)
	a_buf.load_f32(a.to_f32())!
	b_buf.load_f32(b.to_f32())!

	// GEMM on GPU: C = A * B
	vulkan.gemm(dev, c_buf, a_buf, b_buf, m, n, k)!

	// Download result
	c_data := c_buf.download_f32(int(m * n))!
	
	// Convert back to T and reshape
	mut c_tensor := vtl.from_1d[T](c_data.map(vtl.cast[T](it)), vtl.TensorData{})
	return c_tensor.reshape([int(m), int(n)])!
}
