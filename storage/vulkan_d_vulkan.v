module storage

import vsl.vulkan

// VulkanStorageParams provides parameters for creating VulkanStorage.
// device must be set to an initialised vulkan.Device.
@[params]
pub struct VulkanStorageParams {
pub:
	device &vulkan.Device = unsafe { nil }
}

// new_vulkan_params creates a VulkanStorageParams with the given device.
pub fn new_vulkan_params(dev &vulkan.Device) VulkanStorageParams {
	return VulkanStorageParams{ device: dev }
}

// VulkanStorage wraps a Vulkan GPU buffer as a VTL storage backend.
// Mirrors the VclStorage pattern from storage/vcl_d_vcl.v.
@[heap]
pub struct VulkanStorage[T] {
pub mut:
	data   &vulkan.GpuBuffer
	nelems int
}

// vulkan converts CPU storage to a Vulkan GPU buffer.
// params.device must be a valid, initialised vulkan.Device.
pub fn (cpu &CpuStorage[T]) vulkan(params VulkanStorageParams) !&VulkanStorage[T] {
	if isnil(params.device) {
		return error('VulkanStorageParams: device must be set (call vulkan.new_device() and pass it)')
	}
	device := params.device

	n := cpu.data.len
	esz := usize(sizeof(T))
	size := vulkan.DeviceSize(usize(n) * esz)

	mut buf := device.buffer(size)!
	mut bytes := []u8{len: int(size)}
	unsafe { C.memcpy(bytes.data, cpu.data.data, usize(size)) }
	buf.load(bytes)!

	return &VulkanStorage[T]{
		data:   buf
		nelems: n
	}
}

// cpu reads data from GPU back to CPU storage.
pub fn (s &VulkanStorage[T]) cpu() !&CpuStorage[T] {
	arr := s.to_array()!
	return &CpuStorage[T]{
		data: arr
	}
}

// to_array reads data from GPU and returns it as a []T.
@[inline]
pub fn (s &VulkanStorage[T]) to_array[T]() ![]T {
	mut bytes := []u8{len: int(s.data.size)}
	bytes = s.data.store(mut bytes) or { return error('VulkanStorage.to_array: store failed: ${err}') }
	mut result := []T{len: s.nelems}
	unsafe { C.memcpy(result.data, bytes.data, bytes.len) }
	return result
}

// release frees the GPU buffer.
@[inline]
pub fn (s &VulkanStorage[T]) release() ! {
	if !isnil(s.data) {
		s.data.release()
	}
}
