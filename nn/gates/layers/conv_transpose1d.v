module layers

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct ConvTranspose1DGate[T] {
pub:
	input  &autograd.Variable[T] = unsafe { nil }
	weight &autograd.Variable[T] = unsafe { nil }
	bias   &autograd.Variable[T] = unsafe { nil }
	config internal.ConvTranspose1DConfig
}

pub fn conv_transpose1d_gate[T](input &autograd.Variable[T], weight &autograd.Variable[T], bias &autograd.Variable[T], config internal.ConvTranspose1DConfig) &ConvTranspose1DGate[T] {
	return &ConvTranspose1DGate[T]{ input: input, weight: weight, bias: bias, config: config }
}

pub fn (g &ConvTranspose1DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return internal.conv_transpose1d_backward[T](payload.variable.grad, g.input.value, g.weight.value, g.bias.value, g.config)!
}

fn conv_transpose1d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&ConvTranspose1DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

pub fn (g &ConvTranspose1DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	if args.len != 3 {
		return error('ConvTranspose1DGate.cache: expected input, weight, and bias variables')
	}
	match args[0] {
		autograd.Variable[T] {
			match args[1] {
				autograd.Variable[T] {
					match args[2] {
						autograd.Variable[T] {
							autograd.register[T]('ConvTranspose1D', voidptr(g),
								conv_transpose1d_gate_backward_dispatch[T], result, [
									args[0],
									args[1],
									args[2],
								])!
						}
						else {
							return error('ConvTranspose1DGate: bias must be a Variable')
						}
					}
				}
				else {
					return error('ConvTranspose1DGate: weight must be a Variable')
				}
			}
		}
		else {
			return error('ConvTranspose1DGate: input must be a Variable')
		}
	}
}
