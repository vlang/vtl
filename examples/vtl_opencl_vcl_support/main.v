import vsl.vcl
import vsl.vcl.compute
import vtl

fn main() {
	devices := vcl.get_devices(.all) or { panic(err) }
	if devices.len == 0 {
		println('SKIP: no OpenCL devices are available')
		return
	}
	defer {
		for device in devices {
			device.release() or { panic(err) }
		}
	}

	mut device := devices[0]
	values := [1.0, 2.0, 3.0]
	tensor := vtl.from_1d(values)!
	device_tensor := tensor.vcl(device: device)!
	defer {
		device_tensor.release() or { panic(err) }
	}
	round_trip := device_tensor.cpu()!
	if round_trip.data.data != values {
		panic('VTL OpenCL transfer returned unexpected values: ${round_trip.data.data}')
	}

	result := compute.add_scalar_vcl(mut device, values, 2.0)!
	expected := [3.0, 4.0, 5.0]
	if result != expected {
		panic('OpenCL add_scalar returned ${result}; expected ${expected}')
	}
	println('VTL OpenCL tensor transfer and VCL add_scalar passed on ${device}')
}
