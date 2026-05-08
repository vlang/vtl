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
		w_ih := ctx.variable(internal.kaiming_normal[T](3 * hidden_size, input_size, .relu), true)
		w_hh := ctx.variable(internal.kaiming_normal[T](3 * hidden_size, hidden_size, .relu), true)
		b_ih := ctx.variable(vtl.zeros[T](3 * hidden_size), true)
		b_hh := ctx.variable(vtl.zeros[T](3 * hidden_size), true)

		mut layer := &LSTMLayer[T]{
			w_ih: w_ih
			w_hh: w_hh
			b_ih: b_ih
			b_hh: b_hh
			ctx: ctx
			hidden_size: hidden_size
			num_layers: num_layers
		}
		return types.layer[T](layer)
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
		seq_len := input.value.shape[0]
		batch := input.value.shape[1]

		hidden0 := l.ctx.variable(vtl.zeros[T](batch, l.hidden_size), true)
		cell0 := l.ctx.variable(vtl.zeros[T](batch, l.hidden_size), true)

		output, _ := internal.lstm_forward_single[T](
			input.value,
			hidden0.value,
			l.w_ih.value,
			l.w_hh.value,
			l.b_ih.value,
		)!

		return l.ctx.variable(output, false)
	}
}