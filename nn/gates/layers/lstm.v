module layers

import vtl
import vtl.autograd
import vtl.internal

pub struct LSTMGate[T] {
mut:
	input_hidden &autograd.Variable[T]
	hidden        &autograd.Variable[T]
	cell          &autograd.Variable[T]
	w_ih          &autograd.Variable[T]
	w_hh          &autograd.Variable[T]
pub:
	output_hidden &autograd.Variable[T]
	output_cell   &autograd.Variable[T]
}

fn (g &LSTMGate[T]) backward(dout &vtl.Tensor[T]) ! {
}

pub fn lstm_gate[T](ctx &autograd.Context[T], input_hidden &autograd.Variable[T], hidden &autograd.Variable[T], cell &autograd.Variable[T], w_ih &autograd.Variable[T], w_hh &autograd.Variable[T], b_ih &autograd.Variable[T], b_hh &autograd.Variable[T]) !LSTMGate[T] {
	unsafe {
		whh_t := w_hh.value.trT(.none)!
		ih_t := w_ih.value.trT(.none)!

		gates := vtl.add[T](
			vtl.matmul[T](input_hidden.value, ih_t)!,
			vtl.matmul[T](hidden.value, whh_t)!
		)!
		gates = vtl.add[T](gates, b_ih.value)!
		gates = vtl.add[T](gates, b_hh.value)!

		hidden_size := w_ih.value.shape[0] / 3

		i_idx := []int{0, hidden_size}
		f_idx := []int{hidden_size, 2 * hidden_size}
		o_idx := []int{2 * hidden_size, 3 * hidden_size}
		c_idx := []int{0, 3 * hidden_size}

		i_t := internal.sigmoid(ctx, vtl.slice[T](gates, i_idx[0], i_idx[1])!)
		f_t := internal.sigmoid(ctx, vtl.slice[T](gates, f_idx[0], f_idx[1])!)
		o_t := internal.sigmoid(ctx, vtl.slice[T](gates, o_idx[0], o_idx[1])!)
		c_tilde := internal.tanh(ctx, vtl.slice[T](gates, c_idx[0], c_idx[1])!)

		f_cell := vtl.mul[T](f_t, cell.value)!
		i_c := vtl.mul[T](i_t, c_tilde)!
		cell_new := vtl.add[T](f_cell, i_c)!

		h_o := internal.tanh(ctx, cell_new)!
		hidden_new := vtl.mul[T](o_t, h_o)!

		return LSTMGate[T]{
			input_hidden: input_hidden
			hidden: hidden
			cell: cell
			w_ih: w_ih
			w_hh: w_hh
			output_hidden: ctx.variable(hidden_new, false)
			output_cell: ctx.variable(cell_new, false)
		}
	}
}