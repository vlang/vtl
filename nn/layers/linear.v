module layers

import vtl
import vtl.la
import vtl.autograd

pub struct LinearGate<T> {
pub:
        input &autograd.Variable<T>
        weight &autograd.Variable<T>
        bias &autograd.Variable<T>
}

pub fn (g &LinearGate<T>) backward<T>(payload &Payload<T>) []&vtl.Tensor<T> {
        grad := payload.variable.grad
        mut result := [grad, grad, grad]

        if input.requires_grad {
                result[0] = la.matmul(grad, weight.value)
        }

        if weight.requires_grad {
                result[1] = la.matmul(grad.t(), input.value)
        }

        if bias.requires_grad {
                result[2] = la.sum(grad, 0)
        }

	return result
}

pub fn (g &LinearGate<T>) cache<T>(mut result Variable<T>, args ...autograd.CacheParam) {
	a := args[0]
	b := args[1]

	if a is Variable<T> && b is Variable<T> {
		result.grad = vtl.zeros_like<T>(result.value)
		result.requires_grad = true

		register<T>('Divide', g, result, a, b)
	}
}
