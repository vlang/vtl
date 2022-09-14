module loss

import vtl
import vtl.autograd
import vtl.nn.types

pub fn loss_loss<T>(loss types.Loss, input &autograd.Variable<T>, target &vtl.Tensor<T>) ?&autograd.Variable<T> {
	match loss {
		MSELoss<T> {
			return loss.loss(input, target)
		}
		SigmoidCrossEntropyLoss<T> {
			return loss.loss(input, target)
		}
		SoftmaxCrossEntropyLoss<T> {
			return loss.loss(input, target)
		}
		else {
			return error('Loss not implemented for type ${typeof(loss).name}')
		}
	}
}
