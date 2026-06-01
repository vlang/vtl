module layers

import vtl
import vtl.autograd_cuda

pub fn linear_forward_f64_session(input &vtl.Tensor[f64], weights &vtl.Tensor[f64], bias &vtl.Tensor[f64],
	input_gpu voidptr, session voidptr) !&vtl.Tensor[f64] {
	if session == unsafe { nil } {
		return error('linear_forward_f64_session: nil session')
	}
	mut s := unsafe { &autograd_cuda.DeviceSession(session) }
	return s.linear_forward_f64(input, weights, bias, input_gpu)
}
