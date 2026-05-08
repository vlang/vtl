module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types

// BatchNorm1D normalizes a 2D input [batch, features].
// Tracks running mean and variance for inference.
@[params]
pub struct BatchNorm1DConfig {
	eps      f64  = 1e-5
	momentum f64  = 0.1
	affine   bool = true
}

pub struct BatchNorm1DLayer[T] {
	eps      f64
	momentum f64
pub mut:
	gamma               &autograd.Variable[T] = unsafe { nil }
	beta                &autograd.Variable[T] = unsafe { nil }
	running_mean        &vtl.Tensor[T]        = unsafe { nil }
	running_var         &vtl.Tensor[T]        = unsafe { nil }
	num_batches_tracked int
}

pub fn batchnorm1d_layer[T](ctx &autograd.Context[T], num_features int, config BatchNorm1DConfig) types.Layer[T] {
	gamma := ctx.variable(vtl.ones[T]([1, num_features]))
	beta := ctx.variable(vtl.zeros[T]([1, num_features]))
	return types.Layer[T](&BatchNorm1DLayer[T]{
		eps:          config.eps
		momentum:     config.momentum
		gamma:        gamma
		beta:         beta
		running_mean: vtl.zeros[T]([1, num_features])
		running_var:  vtl.ones[T]([1, num_features])
	})
}

pub fn (layer &BatchNorm1DLayer[T]) output_shape() []int {
	return [layer.gamma.value.shape[1]]
}

pub fn (layer &BatchNorm1DLayer[T]) variables() []&autograd.Variable[T] {
	return [layer.gamma, layer.beta]
}

pub fn (layer &BatchNorm1DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	if !input.requires_grad {
		// Inference path: use running stats
		output := internal.batchnorm1d_forward[T](input.value, layer.gamma.value, layer.beta.value,
			layer.running_mean, layer.running_var, layer.eps)!
		return input.context.variable(output)
	}
	// Training path: compute batch stats
	output, batch_mean, batch_var := internal.batchnorm1d_training[T](input.value,
		layer.gamma.value, layer.beta.value, layer.eps)!

	// Update running stats element-wise using scalar arithmetic
	// Use unsafe pointer cast to mutate through immutable receiver
	num_features := layer.running_mean.shape[1]
	momentum := layer.momentum
	mut rm := unsafe { layer.running_mean }
	mut rv := unsafe { layer.running_var }
	for c in 0 .. num_features {
		old_mean := f64(rm.get([0, c]))
		new_mean := momentum * f64(batch_mean.get([0, c])) + (1.0 - momentum) * old_mean
		rm.set([0, c], vtl.cast[T](new_mean))

		old_var := f64(rv.get([0, c]))
		new_var := momentum * f64(batch_var.get([0, c])) + (1.0 - momentum) * old_var
		rv.set([0, c], vtl.cast[T](new_var))
	}

	mut result := input.context.variable(output)
	gate := batchnorm1d_gate[T](input.value, layer.gamma.value, layer.beta.value, batch_mean,
		batch_var, layer.eps)
	gate.cache(mut result, input)!
	return result
}

pub struct BatchNorm1DGate[T] {
	input &vtl.Tensor[T] = unsafe { nil }
	gamma &vtl.Tensor[T] = unsafe { nil }
	beta  &vtl.Tensor[T] = unsafe { nil }
	mean  &vtl.Tensor[T] = unsafe { nil }
	var_  &vtl.Tensor[T] = unsafe { nil }
	eps   f64
}

pub fn batchnorm1d_gate[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], mean &vtl.Tensor[T], var_ &vtl.Tensor[T], eps f64) &BatchNorm1DGate[T] {
	return &BatchNorm1DGate[T]{
		input: input
		gamma: gamma
		beta:  beta
		mean:  mean
		var_:  var_
		eps:   eps
	}
}

pub fn (g &BatchNorm1DGate[T]) backward[T](payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return internal.batchnorm1d_backward[T](payload.variable.grad, g.input, g.gamma, g.beta,
		g.mean, g.var_, g.eps)
}

pub fn (g &BatchNorm1DGate[T]) cache[T](mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	a := args[0]
	match a {
		autograd.Variable[T] {
			result.grad = vtl.zeros_like[T](result.value)
			result.requires_grad = true
			autograd.register[T]('BatchNorm1D', g, result, [a])!
		}
		else {}
	}
}
