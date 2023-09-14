module layers

import vtl
import vtl.autograd
import vtl.la
import vtl.stats

pub struct LinearGate[T] {
pub:
	input  &autograd.Variable[T]
	weight &autograd.Variable[T]
	bias   &autograd.Variable[T]
}

pub fn linear_gate[T](input &autograd.Variable[T], weight &autograd.Variable[T], bias &autograd.Variable[T]) &LinearGate[T] {
	return &LinearGate[T]{
		input: input
		weight: weight
		bias: bias
	}
}

pub fn (g &LinearGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	grad := payload.variable.grad
	mut result := [grad, grad, grad]

	if g.input.requires_grad {
		result[0] = la.matmul[T](grad, g.weight.value)!
	}

	if g.weight.requires_grad {
		result[1] = la.matmul[T](grad.t()!, g.input.value)!
	}

	if g.bias.requires_grad {
		result[2] = vtl.from_1d[T]([stats.sum_axis[T](grad, axis: 0)])!
	}

	return result
}

pub fn (g &LinearGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
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

							mut input_ := unsafe { input }
							mut weight_ := unsafe { weight }
							mut bias_ := unsafe { bias }
							autograd.register[T]('Linear', g, result, [input_, weight_, bias_])!
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
