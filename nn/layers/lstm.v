module layers

import vtl.nn.internal
import vtl.nn.types
import vtl.autograd
import vtl

pub struct LSTMLayer[T] {
mut:
	w_ih        &autograd.Variable[T]
	w_hh        &autograd.Variable[T]
	b_ih        &autograd.Variable[T]
	b_hh        &autograd.Variable[T]
pub:
	ctx         &autograd.Context[T]
	hidden_size int
	num_layers  int
}

pub fn lstm_layer[T](ctx &autograd.Context[T], input_size int, hidden_size int, num_layers int) types.Layer[T] {
	unsafe {
		w_ih_shape := [3 * hidden_size, input_size]
		w_hh_shape := [3 * hidden_size, hidden_size]
		b_shape := [3 * hidden_size]
		w_ih := ctx.variable(internal.kaiming_normal[T](w_ih_shape))
		w_hh := ctx.variable(internal.kaiming_normal[T](w_hh_shape))
		b_ih := ctx.variable(vtl.zeros[T](b_shape))
		b_hh := ctx.variable(vtl.zeros[T](b_shape))

		mut layer := &LSTMLayer[T]{
			w_ih: w_ih
			w_hh: w_hh
			b_ih: b_ih
			b_hh: b_hh
			ctx: ctx
			hidden_size: hidden_size
			num_layers: num_layers
		}
		return types.Layer[T](layer)
	}
}

fn (l &LSTMLayer[T]) output_shape() []int {
	return [l.hidden_size]
}

fn (l &LSTMLayer[T]) variables() []&autograd.Variable[T] {
	return [l.w_ih, l.w_hh, l.b_ih, l.b_hh]
}

fn (l &LSTMLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	unsafe {
		batch := input.value.shape[0]

		hidden0 := l.ctx.variable(vtl.zeros[T]([batch, l.hidden_size]))
		cell0 := l.ctx.variable(vtl.zeros[T]([batch, l.hidden_size]))

		output, _ := internal.lstm_forward_single[T](
			input.value,
			hidden0.value,
			l.w_ih.value,
			l.w_hh.value,
			l.b_ih.value,
			l.b_hh.value,
		)!

		return l.ctx.variable(output)
	}
}