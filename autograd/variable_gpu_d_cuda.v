module autograd

import vtl

pub fn (mut v Variable[f64]) clear_gpu_activation() {
	if v.gpu_activation != unsafe { nil } {
		unsafe { &vtl.CudaTensor[f64](v.gpu_activation) }.release()
		v.gpu_activation = unsafe { nil }
	}
}

pub fn (mut v Variable[f64]) set_gpu_activation(t &vtl.CudaTensor[f64]) {
	v.clear_gpu_activation()
	v.gpu_activation = voidptr(t)
}

pub fn (mut v Variable[f64]) take_gpu_activation_input() voidptr {
	if v.gpu_activation == unsafe { nil } {
		return unsafe { nil }
	}
	ptr := v.gpu_activation
	v.gpu_activation = unsafe { nil }
	return ptr
}

pub fn (mut s DeviceSession) bind_gpu_activation_to_variable(mut v Variable[f64]) {
	if !gpu_activations_enabled() {
		return
	}
	t := take_chain_activation(mut s) or { return }
	v.set_gpu_activation(t)
}

fn take_chain_activation(mut s DeviceSession) !&vtl.CudaTensor[f64] {
	return take_chain_activation_impl(mut s)
}
