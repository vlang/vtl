module autograd_cuda

// Stubs when not built with `-d cuda`.

// session_bind_gpu_activation exposes this operation as part of the public API.
pub fn session_bind_gpu_activation(mut s DeviceSession, act_field &voidptr) {}
