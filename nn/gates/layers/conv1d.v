module layers

import vtl
import vtl.autograd
import vtl.nn.internal

// Conv1DGate stores tensors required to differentiate a 1D convolution.
pub struct Conv1DGate[T] {
pub:
	input  &autograd.Variable[T] = unsafe { nil }
	weight &autograd.Variable[T] = unsafe { nil }
	bias   &autograd.Variable[T] = unsafe { nil }
	config internal.Conv1DConfig
}

// conv1d_gate creates an autograd gate for Conv1D.
pub fn conv1d_gate[T](input &autograd.Variable[T], weight &autograd.Variable[T],
	bias &autograd.Variable[T], config internal.Conv1DConfig) &Conv1DGate[T] {
	return &Conv1DGate[T]{ input: input, weight: weight, bias: bias, config: config }
}

// backward returns gradients for input, weight, and bias.
pub fn (g &Conv1DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return internal.conv1d_backward[T](payload.variable.grad, g.input.value, g.weight.value,
		g.bias.value, g.config)!
}

fn conv1d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&Conv1DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache registers the convolution in the autograd graph.
pub fn (g &Conv1DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	if args.len != 3 {
		return error('Conv1DGate.cache: expected input, weight, and bias variables')
	}
	input := args[0]
	weight := args[1]
	bias := args[2]
	match input {
		autograd.Variable[T] {
			match weight {
				autograd.Variable[T] {
					match bias {
						autograd.Variable[T] {
							result.grad = vtl.zeros_like[T](result.value)
							result.requires_grad = true
							autograd.register[T]('Conv1D', voidptr(g),
								conv1d_gate_backward_dispatch[T], result, [input, weight, bias])!
						}
						else {
							return error('Conv1DGate: bias must be a Variable')
						}
					}
				}
				else {
					return error('Conv1DGate: weight must be a Variable')
				}
			}
		}
		else {
			return error('Conv1DGate: input must be a Variable')
		}
	}
}
