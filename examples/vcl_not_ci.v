import vtl
import vsl.vcl

// get all devices if you want
devices := vcl.get_devices(vcl.device_cpu)?
println('Devices: $devices')

// do not create platforms/devices/contexts/queues/...
// just get the device
mut device := vcl.get_default_device()?
defer {
	device.release() or { panic(err) }
}

t := vtl.from_1d([1, 2, 3, 4, 5, 6])?
cl := t.vcl(device: device)?

println(cl)
