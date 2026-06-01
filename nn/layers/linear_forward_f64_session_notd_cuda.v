module layers

import vtl

// linear_forward_f64_session is a no-op without `-d cuda`.
pub fn linear_forward_f64_session(input &vtl.Tensor[f64], weights &vtl.Tensor[f64], bias &vtl.Tensor[f64],
	_input_gpu voidptr, session voidptr) !&vtl.Tensor[f64] {
	return error('linear_forward_f64_session: build without -d cuda')
}
