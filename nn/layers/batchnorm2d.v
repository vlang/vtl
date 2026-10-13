module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.types

// BatchNorm2D normalizes a 4D input [batch, channels, height, width] per
// channel, following PyTorch's BatchNorm2d. It tracks running mean and
// variance for inference; the running variance is updated with the
// unbiased batch variance, as in PyTorch.

// BatchNorm2DConfig configures a BatchNorm2D layer.
@[params]
pub struct BatchNorm2DConfig {
pub:
	eps      f64  = 1e-5
	momentum f64  = 0.1
	affine   bool = true
}

// BatchNorm2DLayer is the BatchNorm2D layer state.
pub struct BatchNorm2DLayer[T] {
	eps      f64
	momentum f64
	affine   bool
pub mut:
	gamma               &autograd.Variable[T] = unsafe { nil }
	beta                &autograd.Variable[T] = unsafe { nil }
	running_mean        &vtl.Tensor[T]        = unsafe { nil }
	running_var         &vtl.Tensor[T]        = unsafe { nil }
	num_batches_tracked int
}

// batchnorm2d_layer creates a BatchNorm2D layer for `num_features` channels.
pub fn batchnorm2d_layer[T](ctx &autograd.Context[T], num_features int, config BatchNorm2DConfig) types.Layer[T] {
	gamma := ctx.variable(vtl.ones[T]([num_features]), requires_grad: config.affine)
	beta := ctx.variable(vtl.zeros[T]([num_features]), requires_grad: config.affine)
	layer := &BatchNorm2DLayer[T]{
		eps:          config.eps
		momentum:     config.momentum
		affine:       config.affine
		gamma:        gamma
		beta:         beta
		running_mean: vtl.zeros[T]([num_features])
		running_var:  vtl.ones[T]([num_features])
	}
	return types.layer_with_mode[T](voidptr(layer), batch_norm2_d_layer_output_shape_dispatch[T],
		batch_norm2_d_layer_variables_dispatch[T], batch_norm2_d_layer_forward_dispatch[T],
		batch_norm2_d_layer_forward_mode_dispatch[T])
}

// output_shape returns the per-channel feature count.
pub fn (layer &BatchNorm2DLayer[T]) output_shape() []int {
	return [layer.gamma.value.shape[0]]
}

// variables returns the affine parameters, or none when affine is disabled.
pub fn (layer &BatchNorm2DLayer[T]) variables() []&autograd.Variable[T] {
	if !layer.affine {
		return []&autograd.Variable[T]{}
	}
	return [layer.gamma, layer.beta]
}

// forward runs the layer; the mode follows the input's gradient requirement,
// mirroring BatchNorm1D. Prefer forward_with_mode through Sequential, which
// propagates train()/eval().
pub fn (layer &BatchNorm2DLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	return layer.forward_with_mode(input, input.requires_grad)
}

// forward_with_mode uses batch statistics and updates running statistics in
// training mode, and uses the stored running statistics in evaluation mode.
pub fn (layer &BatchNorm2DLayer[T]) forward_with_mode(input &autograd.Variable[T], training bool) !&autograd.Variable[T] {
	if input.value.shape.len != 4 {
		return error('BatchNorm2D expects a 4D input [batch, channels, height, width], got shape ${input.value.shape}')
	}
	if input.value.shape[1] != layer.gamma.value.shape[0] {
		return error('BatchNorm2D expected ${layer.gamma.value.shape[0]} channels, got ${input.value.shape[1]}')
	}
	if !training {
		output := internal.batchnorm2d_forward[T](input.value, layer.gamma.value, layer.beta.value,
			layer.running_mean, layer.running_var, layer.eps)!
		mut result := input.context.variable(output,
			requires_grad: input.requires_grad
				|| layer.gamma.requires_grad || layer.beta.requires_grad
		)
		if result.requires_grad {
			gate := batchnorm2d_gate_with_mode[T](input.value, layer.gamma.value, layer.beta.value,
				layer.running_mean, layer.running_var, layer.eps, false)
			gate.cache(mut result, input, layer.gamma, layer.beta)!
		}
		return result
	}

	output, batch_mean, batch_var := internal.batchnorm2d_training[T](input.value,
		layer.gamma.value, layer.beta.value, layer.eps)!

	// Update running statistics through an unsafe pointer cast to mutate
	// through the immutable receiver, mirroring BatchNorm1D.
	channels := layer.running_mean.shape[0]
	momentum := layer.momentum
	count := f64(input.value.shape[0] * input.value.shape[2] * input.value.shape[3])
	mut rm := unsafe { layer.running_mean }
	mut rv := unsafe { layer.running_var }
	for c in 0 .. channels {
		old_mean := f64(rm.get([c]))
		new_mean := momentum * f64(batch_mean.get([c])) + (1.0 - momentum) * old_mean
		rm.set([c], vtl.cast[T](new_mean))

		// PyTorch stores the unbiased variance in the running estimate.
		biased_var := f64(batch_var.get([c]))
		unbiased_var := if count > 1.0 { biased_var * count / (count - 1.0) } else { biased_var }
		old_var := f64(rv.get([c]))
		new_var := momentum * unbiased_var + (1.0 - momentum) * old_var
		rv.set([c], vtl.cast[T](new_var))
	}
	mut result := input.context.variable(output,
		requires_grad: input.requires_grad
			|| layer.gamma.requires_grad || layer.beta.requires_grad
	)
	if result.requires_grad {
		gate := batchnorm2d_gate_with_mode[T](input.value, layer.gamma.value, layer.beta.value, batch_mean,
			batch_var, layer.eps, true)
		gate.cache(mut result, input, layer.gamma, layer.beta)!
	}
	return result
}

fn batch_norm2_d_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&BatchNorm2DLayer[T](layer)).output_shape() }
}

fn batch_norm2_d_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&BatchNorm2DLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn batch_norm2_d_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&BatchNorm2DLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}

fn batch_norm2_d_layer_forward_mode_dispatch[T](layer voidptr, input voidptr, training bool) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&BatchNorm2DLayer[T](layer)).forward_with_mode(typed_input, training)! }
	return voidptr(result)
}

// BatchNorm2DGate is the autograd gate for BatchNorm2D.
pub struct BatchNorm2DGate[T] {
	input    &vtl.Tensor[T] = unsafe { nil }
	gamma    &vtl.Tensor[T] = unsafe { nil }
	beta     &vtl.Tensor[T] = unsafe { nil }
	mean     &vtl.Tensor[T] = unsafe { nil }
	var_     &vtl.Tensor[T] = unsafe { nil }
	eps      f64
	training bool
}

// batchnorm2d_gate_with_mode creates the backward gate for train or eval mode.
pub fn batchnorm2d_gate_with_mode[T](input &vtl.Tensor[T], gamma &vtl.Tensor[T], beta &vtl.Tensor[T], mean &vtl.Tensor[T], var_ &vtl.Tensor[T], eps f64, training bool) &BatchNorm2DGate[T] {
	return &BatchNorm2DGate[T]{
		input:    input
		gamma:    gamma
		beta:     beta
		mean:     mean
		var_:     var_
		eps:      eps
		training: training
	}
}

// backward computes gradients w.r.t. input, gamma and beta.
pub fn (g &BatchNorm2DGate[T]) backward(payload &autograd.Payload[T]) ![]&vtl.Tensor[T] {
	return internal.batchnorm2d_backward_mode[T](payload.variable.grad, g.input, g.gamma, g.beta,
		g.mean, g.var_, g.eps, g.training)
}

fn batch_norm2_d_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &autograd.Payload[T](payload) }
	tensors := unsafe { (&BatchNorm2DGate[T](gate)).backward(typed_payload)! }
	return autograd.tensor_ptrs_to_voidptrs[T](tensors)
}

// cache registers the gate on the result variable.
pub fn (g &BatchNorm2DGate[T]) cache(mut result autograd.Variable[T], args ...autograd.CacheParam) ! {
	input := args[0]
	match input {
		autograd.Variable[T] {
			gamma := args[1]
			match gamma {
				autograd.Variable[T] {
					beta := args[2]
					match beta {
						autograd.Variable[T] {
							autograd.register[T]('BatchNorm2D', voidptr(g),
								batch_norm2_d_gate_backward_dispatch[T], result, [
									input,
									gamma,
									beta,
								])!
						}
						else {
							return error('BatchNorm2DGate: beta must be a Variable')
						}
					}
				}
				else {
					return error('BatchNorm2DGate: gamma must be a Variable')
				}
			}
		}
		else {
			return error('BatchNorm2DGate: input must be a Variable')
		}
	}
}
