module layers

import vtl.nn.internal
import vtl.nn.types
import vtl.autograd
import vtl

// LSTMLayer implements a Long Short-Term Memory layer.
//
// Implements the standard LSTM equations per timestep:
//   i_t = sigmoid(x_t @ W_ii^T + h_{t-1} @ W_hi^T + b_ii + b_hi)
//   f_t = sigmoid(x_t @ W_if^T + h_{t-1} @ W_hf^T + b_if + b_hf)
//   g_t = tanh(x_t @ W_ig^T + h_{t-1} @ W_hg^T + b_ig + b_hg)
//   o_t = sigmoid(x_t @ W_io^T + h_{t-1} @ W_ho^T + b_io + b_ho)
//   c_t = f_t * c_{t-1} + i_t * g_t
//   h_t = o_t * tanh(c_t)
//
// Input:  `[batch, seq_len, input_size]`
// Output: `[batch, hidden_size]`  (final hidden state)
//
// Contains learnable weights:
//   w_ih: [4*hidden_size, input_size]  — input-to-hidden for all 4 gates
//   w_hh: [4*hidden_size, hidden_size] — hidden-to-hidden for all 4 gates
//   b_ih: [4*hidden_size]              — input bias
//   b_hh: [4*hidden_size]              — hidden bias
pub struct LSTMLayer[T] {
mut:
	w_ih &autograd.Variable[T]
	w_hh &autograd.Variable[T]
	b_ih &autograd.Variable[T]
	b_hh &autograd.Variable[T]
pub:
	ctx         &autograd.Context[T]
	input_size  int
	hidden_size int
	num_layers  int
}

// lstm_layer creates an LSTMLayer with Kaiming-initialized weights.
pub fn lstm_layer[T](ctx &autograd.Context[T], input_size int, hidden_size int, num_layers int) types.Layer[T] {
	unsafe {
		w_ih_shape := [4 * hidden_size, input_size]
		w_hh_shape := [4 * hidden_size, hidden_size]
		b_shape := [4 * hidden_size]
		w_ih := ctx.variable(internal.kaiming_normal[T](w_ih_shape))
		w_hh := ctx.variable(internal.kaiming_normal[T](w_hh_shape))
		b_ih := ctx.variable(vtl.zeros[T](b_shape))
		b_hh := ctx.variable(vtl.zeros[T](b_shape))

		mut layer := &LSTMLayer[T]{
			w_ih:        w_ih
			w_hh:        w_hh
			b_ih:        b_ih
			b_hh:        b_hh
			ctx:         ctx
			input_size:  input_size
			hidden_size: hidden_size
			num_layers:  num_layers
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

// forward runs the LSTM over the input sequence.
// Input shape: [batch, seq_len, input_size]
// Internally transposes to [seq_len, batch, input_size] for the step loop.
// Returns the final hidden state: [batch, hidden_size]
fn (l &LSTMLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	unsafe {
		batch := input.value.shape[0]
		// Transpose from [batch, seq_len, input_size] to [seq_len, batch, input_size]
		input_t := input.value.transpose([1, 0, 2])!

		hidden0 := vtl.zeros[T]([batch, l.hidden_size])
		cell0 := vtl.zeros[T]([batch, l.hidden_size])

		output, h_n, _ := internal.lstm_forward_single[T](input_t, hidden0, cell0, l.w_ih.value,
			l.w_hh.value, l.b_ih.value, l.b_hh.value)!

		_ = output // full sequence output — could expose via a separate method
		return l.ctx.variable(h_n)
	}
}
