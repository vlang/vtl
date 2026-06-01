module autograd_cuda

// Stubs when not built with `-d cuda`.

pub fn session_bind_gpu_activation(mut s DeviceSession, act_field &voidptr) {}
