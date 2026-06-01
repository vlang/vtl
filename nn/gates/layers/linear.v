module layers

import vtl
import vtl.autograd
import vtl.la

// LinearGate defines a public data structure for this module.
pub struct LinearGate[T] {
pub:
	input  &autograd.Variable[T] = unsafe { nil }
	weight &autograd.Variable[T] = unsafe { nil }
	bias   &autograd.Variable[T] = unsafe { nil }
}

// linear_gate exposes this operation as part of the public API.
pub fn linear_gate[T](input &autograd.Variable[T], weight &autograd.Variable[T], bias &autograd.Variable[T]) &LinearGate[T] {
	return &LinearGate[T]{
		input:  input
		weight: weight
		bias:   bias
	}
}

// backward exposes this operation as part of the public API.
pub fn (g &LinearGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := payload.variable.grad
	mut result := [grad, grad, grad]
	$if sizeof(T) == 8 {
		$if cuda ? {
			if linear_gate_use_cuda_backward() {
				tensors := linear_gate_backward_f64_cuda(voidptr(g), voidptr(payload))!
				if g.input.requires_grad {
					result[0] = unsafe { &vtl.Tensor[T](tensors[0]) }
				}
				if g.weight.requires_grad {
					result[1] = unsafe { &vtl.Tensor[T](tensors[1]) }
				}
				if g.bias.requires_grad {
					result[2] = unsafe { &vtl.Tensor[T](tensors[2]) }
				}
				return result
			}
		}
		if g.input.requires_grad {
			result[0] = la.matmul[f64](grad, g.weight.value)!
		}
		if g.weight.requires_grad {
			result[1] = la.matmul[f64](grad.t()!, g.input.value)!
		}
		if g.bias.requires_grad {
			batch_size := grad.shape[0]
			ones := vtl.ones[f64]([1, batch_size])
			result[2] = la.matmul[f64](ones, grad)!
		}
		return result
	} $else $if sizeof(T) == 4 {
		if tensors := linear_gate_backward_f32_try(voidptr(g), voidptr(payload)) {
			if g.input.requires_grad {
				result[0] = tensors[0]
			}
			if g.weight.requires_grad {
				result[1] = tensors[1]
			}
			if g.bias.requires_grad {
				result[2] = tensors[2]
			}
			return result
		}
		if g.input.requires_grad {
			result[0] = la.matmul[f32](grad, g.weight.value)!
		}
		if g.weight.requires_grad {
			result[1] = la.matmul[f32](grad.t()!, g.input.value)!
		}
		if g.bias.requires_grad {
			batch_size := grad.shape[0]
			ones := vtl.ones[f32]([1, batch_size])
			result[2] = la.matmul[f32](ones, grad)!
		}
		return result
	} $else {
		if g.input.requires_grad {
			result[0] = la.matmul[T](grad, g.weight.value)!
		}
		if g.weight.requires_grad {
			result[1] = la.matmul[T](grad.t()!, g.input.value)!
		}
		if g.bias.requires_grad {
			batch_size := grad.shape[0]
			ones := vtl.ones[T]([1, batch_size])
			result[2] = la.matmul[T](ones, grad)!
		}
		return result
	}
}

fn linear_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&LinearGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache exposes this operation as part of the public API.
pub fn (g &LinearGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
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

							autograd.register[T]('Linear', voidptr(g),
								linear_gate_backward_dispatch[T], result, [input, weight, bias])!
						}
						else {
							return error('LinearGate: bias must be a Variable')
						}
					}
				}
				else {
					return error('LinearGate: weight must be a Variable')
				}
			}
		}
		else {
			return error('LinearGate: input must be a Variable')
		}
	}
}
