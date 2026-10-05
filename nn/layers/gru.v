module layers

import vtl
import vtl.autograd
import vtl.nn.gates.layers
import vtl.nn.internal
import vtl.nn.types

// GRULayer is a single-layer, unidirectional GRU with zero initial state.
// Input and output use [sequence, batch, features] layout.
pub struct GRULayer[T] {
pub mut:
	w_ih &autograd.Variable[T]
	w_hh &autograd.Variable[T]
	b_ih &autograd.Variable[T]
	b_hh &autograd.Variable[T]
pub:
	ctx         &autograd.Context[T]
	input_size  int
	hidden_size int
}

// gru_layer creates a GRU layer with reset, update, and candidate gates.
pub fn gru_layer[T](ctx &autograd.Context[T], input_size int, hidden_size int) types.Layer[T] {
	if input_size <= 0 || hidden_size <= 0 {
		panic('gru_layer: input_size and hidden_size must be positive')
	}
	w_ih := ctx.variable(internal.kaiming_normal[T]([3 * hidden_size, input_size]))
	w_hh := ctx.variable(internal.kaiming_normal[T]([3 * hidden_size, hidden_size]))
	b_ih := ctx.variable(vtl.zeros[T]([3 * hidden_size]))
	b_hh := ctx.variable(vtl.zeros[T]([3 * hidden_size]))
	layer := &GRULayer[T]{
		w_ih:        w_ih
		w_hh:        w_hh
		b_ih:        b_ih
		b_hh:        b_hh
		ctx:         ctx
		input_size:  input_size
		hidden_size: hidden_size
	}
	return types.layer[T](voidptr(layer), gru_layer_output_shape_dispatch[T],
		gru_layer_variables_dispatch[T], gru_layer_forward_dispatch[T])
}

fn (l &GRULayer[T]) output_shape() []int { return [l.hidden_size] }

fn (l &GRULayer[T]) variables() []&autograd.Variable[T] {
	return [l.w_ih, l.w_hh, l.b_ih, l.b_hh]
}

fn (l &GRULayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	if input.context != l.ctx {
		return error('GRULayer.forward: input and layer must use the same autograd context')
	}
	if input.value.shape.len != 3 || input.value.shape[2] != l.input_size {
		return error('GRULayer.forward: input must have shape [sequence, batch, ${l.input_size}]')
	}
	batch := input.value.shape[1]
	h0 := vtl.zeros[T]([batch, l.hidden_size])
	output, _ := internal.gru_forward_single[T](input.value, l.w_ih.value, l.w_hh.value,
		l.b_ih.value, l.b_hh.value, h0)!
	mut result := l.ctx.variable(output)
	if input.is_grad_needed() || l.w_ih.is_grad_needed() || l.w_hh.is_grad_needed()
		|| l.b_ih.is_grad_needed() || l.b_hh.is_grad_needed() {
		gate := layers.gru_gate[T](input, l.w_ih, l.w_hh, l.b_ih, l.b_hh)
		gate.cache(mut result)!
	}
	return result
}

fn gru_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&GRULayer[T](layer)).output_shape() }
}

fn gru_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&GRULayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn gru_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&GRULayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
