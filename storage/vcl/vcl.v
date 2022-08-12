module vcl

import vsl.vcl

// VclStorage
[heap]
pub struct VclStorage<T> {
pub mut:
	data vcl.Vector<T>
}

pub fn new_storage<T>(device &vcl.Device, len int, init T) ?&VclStorage<T> {
	mut data := device.vector<T>(len)?
	if init != T(0) {
		init_data := []T{len: len, init: init}
		data.load(init_data)
	}
	return &VclStorage<T>{
		data: data
	}
}

pub fn from_array<T>(device &vcl.Device, arr []T) ?&VclStorage<T> {
	mut data := device.vector<T>(len)?
	data.load(arr.clone())
	return &VclStorage<T>{
		data: data
	}
}

[inline]
pub fn (storage &VclStorage<T>) to_array<T>() []T {
	return storage.data.data() or { []T{} }
}
