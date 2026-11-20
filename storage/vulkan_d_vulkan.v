module storage

import vsl.vulkan

// VulkanParams defines a public data structure for this module.

// VulkanParams defines a public data structure for this module.
@[params]
pub struct VulkanParams {
pub mut:
	device &vulkan.Device = unsafe { nil }
}

// vulkan_params_for_device builds params so all uploads share one Vulkan device.
pub fn vulkan_params_for_device(device &vulkan.Device) VulkanParams {
	return VulkanParams{
		device: device
	}
}

// VulkanStorage defines a public data structure for this module.

// VulkanStorage defines a public data structure for this module.
@[heap]
pub struct VulkanStorage[T] {
pub mut:
	data   &vulkan.GpuBuffer = unsafe { nil }
	length int
}

// vulkan exposes this operation as part of the public API.
pub fn (cpu &CpuStorage[T]) vulkan(params VulkanParams) !&VulkanStorage[T] {
	mut device := params.device
	owns_device := isnil(device)

	if owns_device {
		device = vulkan.new_device()!
	}

	arr := cpu.data
	size := u64(arr.len) * u64(sizeof(T))
	mut buf := device.buffer(vulkan.DeviceSize(size)) or {
		allocation_error := err
		if owns_device {
			device.release() or {
				return error('${allocation_error}; additionally failed to release owned Vulkan device: ${err}')
			}
		}
		return allocation_error
	}
	mut raw := []u8{len: int(size)}
	unsafe { C.memcpy(raw.data, arr.data, size) }
	buf.load(raw) or {
		upload_error := err
		buf.release()
		if owns_device {
			device.release() or {
				return error('${upload_error}; additionally failed to release owned Vulkan device: ${err}')
			}
		}
		return upload_error
	}
	return &VulkanStorage[T]{
		data:   buf
		length: arr.len
	}
}

// vulkan exposes this operation as part of the public API.

// vulkan exposes this operation as part of the public API.
@[inline]
pub fn (vstorage &VulkanStorage[T]) vulkan(params VulkanParams) !&VulkanStorage[T] {
	return vstorage
}

// cpu exposes this operation as part of the public API.
pub fn (vstorage &VulkanStorage[T]) cpu() !&CpuStorage[T] {
	size := u64(vstorage.length) * u64(sizeof(T))
	mut raw := []u8{len: int(size)}
	vstorage.data.store(mut raw)!
	mut arr := []T{len: vstorage.length}
	unsafe { C.memcpy(arr.data, raw.data, size) }
	return &CpuStorage[T]{
		data: arr
	}
}

// to_array exposes this operation as part of the public API.
pub fn (vstorage &VulkanStorage[T]) to_array() ![]T {
	size := u64(vstorage.length) * u64(sizeof(T))
	mut raw := []u8{len: int(size)}
	vstorage.data.store(mut raw)!
	mut arr := []T{len: vstorage.length}
	unsafe { C.memcpy(arr.data, raw.data, size) }
	return arr
}

// release exposes this operation as part of the public API.
pub fn (vstorage &VulkanStorage[T]) release() {
	vstorage.data.release()
}
