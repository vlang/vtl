module autograd_cuda

import vtl.autograd

// attach_context_session initializes Context[f64].device_session when nil.
pub fn attach_context_session(mut ctx &autograd.Context[f64]) {
	if ctx.device_session == unsafe { nil } {
		ctx.device_session = new_device_session_ptr()
	}
}
