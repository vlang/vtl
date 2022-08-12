module storage

import vsl.vcl

[params]
pub struct VclStorageParams {
	device &vcl.Device = unsafe { nil }
}

// VclStorage
[heap]
pub struct VclStorage<T> {
pub mut:
	data vcl.Vector<T>
}

pub fn (cpu &CpuStorage<T>) vcl(params VclStorageParams) ?&VclStorage<T> {
	mut device := params.device

	if isnil(device) {
		device = vcl.get_default_device()?
	}

	arr := cpu.data.clone()
	mut data := device.vector<T>(arr.len)?
	data.load(arr)
	return &VclStorage<T>{
		data: data
	}
}

[inline]
pub fn (storage &VclStorage<T>) to_array<T>() []T {
	return storage.data.data() or { []T{} }
}
