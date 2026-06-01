module layers

import vtl.autograd

// linear_take_gpu_input moves a pending GPU activation from the input Variable (f64, `-d cuda` only).
pub fn linear_take_gpu_input(input voidptr) voidptr {
	mut v := unsafe { &autograd.Variable[f64](input) }
	return v.take_gpu_activation_input()
}
