// Records activations and gradients of a ReLU layer with module hooks.
//
// The layer is never modified: with_hooks wraps it, and the hooks only observe the values
// flowing through the module (and, for the backward hook, the gradient leaving it).
module main

import vtl
import vtl.autograd
import vtl.nn.hooks
import vtl.nn.layers

fn print_activation(input &autograd.Variable[f64], output &autograd.Variable[f64]) !&autograd.Variable[f64] {
	println('forward  input  = ${input.value.to_array()}')
	println('forward  output = ${output.value.to_array()}')
	return output
}

fn print_gradient(grad &vtl.Tensor[f64]) !&vtl.Tensor[f64] {
	println('backward grad at module output = ${grad.to_array()}')
	return grad
}

fn main() {
	ctx := autograd.ctx[f64]()
	relu := layers.relu_layer[f64](ctx, [4])
	hooked, recorder := hooks.with_hooks[f64](relu)

	forward_handle := recorder.add_forward_hook(print_activation)
	backward_handle := recorder.add_backward_hook(print_gradient)

	input := ctx.variable(vtl.from_array([f64(-1), 0.5, 2, -3], [4])!)
	output := hooked.forward(input)!
	output.backprop()!
	println('gradient at the input = ${input.grad.to_array()}')

	// Removing a hook stops it from running on later passes; the layer itself is unchanged.
	forward_handle.remove()
	backward_handle.remove()
	again := ctx.variable(vtl.from_array([f64(1), -1, 0, 4], [4])!)
	mut again_output := hooked.forward(again)!
	again_output.backprop()!
	println('after removing the hooks, no further lines are printed for this pass')
}
