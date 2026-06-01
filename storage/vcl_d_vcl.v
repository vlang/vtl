module storage

import vsl.vcl

// VclStorageParams defines a public data structure for this module.

// VclStorageParams defines a public data structure for this module.
@[params]
pub struct VclStorageParams {
	device &vcl.Device = unsafe { nil }
}

// VclStorage

// VclStorage defines a public data structure for this module.

// VclStorage defines a public data structure for this module.
@[heap]
pub struct VclStorage[T] {
pub mut:
	data vcl.Vector[T]
}

// vcl exposes this operation as part of the public API.
pub fn (cpu &CpuStorage[T]) vcl(params VclStorageParams) !&VclStorage[T] {
	mut device := params.device

	if isnil(device) {
		device = vcl.get_default_device()!
	}

	arr := cpu.data.clone()
	mut data := device.vector[T](arr.len)!
	err := <-data.load(arr)
	if err !is none {
		return err
	}
	return &VclStorage[T]{
		data: data
	}
}

// cpu exposes this operation as part of the public API.
pub fn (storage &VclStorage[T]) cpu() !&CpuStorage[T] {
	arr := storage.to_array()!
	return &CpuStorage[T]{
		data: arr
	}
}

// to_array exposes this operation as part of the public API.

// to_array exposes this operation as part of the public API.
@[inline]
pub fn (storage &VclStorage[T]) to_array[T]() ![]T {
	return storage.data.data()
}

// release exposes this operation as part of the public API.

// release exposes this operation as part of the public API.
@[inline]
pub fn (storage &VclStorage[T]) release() ! {
	return storage.data.release()
}
