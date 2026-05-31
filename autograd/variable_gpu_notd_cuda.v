module autograd

// Stubs when not built with `-d cuda`.

pub fn (mut v Variable[f64]) clear_gpu_activation() {}

pub fn (mut v Variable[f64]) set_gpu_activation(_ voidptr) {}

pub fn (mut v Variable[f64]) take_gpu_activation_input() voidptr {
	return unsafe { nil }
}

pub fn (mut s DeviceSession) bind_gpu_activation_to_variable(mut v Variable[f64]) {
	_ = v
}
