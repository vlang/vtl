module vcl

import vsl.vcl

fn test_vcl_with_default() {
	// get all devices if you want
	devices := vcl.get_devices(vcl.device_cpu) or { panic(err) }
	println('Devices: $devices')

	// do not create platforms/devices/contexts/queues/...
	// just get the device
	mut device := vcl.get_default_device() or { panic(err) }
	defer {
		device.release() or { panic(err) }
	}

	s := new_storage<f64>(device, 2, 1.0) or { panic(err) }
	array := s.to_array()
	assert array.len == 2
	assert array[0] == 1.0
}
