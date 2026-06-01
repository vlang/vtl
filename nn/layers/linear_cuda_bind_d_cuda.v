module layers

import vtl.autograd
import vtl.autograd_cuda

// linear_bind_output_gpu exposes this operation as part of the public API.
pub fn linear_bind_output_gpu(result voidptr) {
	if result == unsafe { nil } {
		return
	}
	mut v := unsafe { &autograd.Variable[f64](result) }
	mut session := unsafe { &autograd_cuda.DeviceSession(v.context.device_session) }
	autograd_cuda.session_bind_gpu_activation(mut session,
		autograd.variable_gpu_activation_ptr(mut v))
}
