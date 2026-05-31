module autograd

import vtl

// init_device is a no-op without `-d cuda`.
pub fn (mut s DeviceSession) init_device() {}

// linear_forward_f64 without CUDA build always errors so callers fall back to CPU.
pub fn (mut s DeviceSession) linear_forward_f64(x &vtl.Tensor[f64], weights &vtl.Tensor[f64], bias &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	return error('device session: build without -d cuda')
}
