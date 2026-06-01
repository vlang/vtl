module autograd_cuda

import vtl

// session_bind_gpu_activation exposes this operation as part of the public API.
pub fn session_bind_gpu_activation(mut s DeviceSession, act_field &voidptr) {
	if !gpu_activations_enabled() || act_field == unsafe { nil } {
		return
	}
	t := take_chain_activation(mut s) or { return }
	if *act_field != unsafe { nil } {
		unsafe { &vtl.CudaTensor[f64](*act_field) }.release()
	}
	unsafe {
		*act_field = voidptr(t)
	}
}

fn take_chain_activation(mut s DeviceSession) !&vtl.CudaTensor[f64] {
	return take_chain_activation_impl(mut s)
}
