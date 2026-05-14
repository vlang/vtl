module layers

import vtl.nn.internal
import vtl.nn.types
import vtl.autograd
import vtl

// GRULayer implements a Gated Recurrent Unit layer.
//
// Implements the standard GRU equations (PyTorch/CuDNN compatible):
//   r_t = sigmoid(x_t @ W_ir^T + h_{t-1} @ W_hr^T + b_ir + b_hr)
//   z_t = sigmoid(x_t @ W_iz^T + h_{t-1} @ W_hz^T + b_iz + b_hz)
//   n_t = tanh(x_t @ W_in^T + b_in + r_t * (h_{t-1} @ W_hn^T + b_hn))
//   h_t = (1 - z_t) * n_t + z_t * h_{t-1}
//
// Input:  `[batch, seq_len, input_size]`
// Output: `[batch, hidden_size]`  (final hidden state)
//
// Compared to LSTM, GRU:
//   - Has 3 gates instead of 4 (no separate cell state)
//   - Uses fewer parameters (3*hidden_size vs 4*hidden_size weights)
//   - Often trains faster while achieving comparable performance
//
// Contains learnable weights:
//   w_ih: [3*hidden_size, input_size]  — input-to-hidden for r, z, n gates
//   w_hh: [3*hidden_size, hidden_size] — hidden-to-hidden for r, z, n gates
//   b_ih: [3*hidden_size]              — input bias
//   b_hh: [3*hidden_size]              — hidden bias
pub struct GRULayer[T] {
mut:
	w_ih &autograd.Variable[T]
	w_hh &autograd.Variable[T]
	b_ih &autograd.Variable[T]
	b_hh &autograd.Variable[T]
pub:
	ctx         &autograd.Context[T]
	input_size  int
	hidden_size int
}

// gru_layer creates a GRULayer with Kaiming-initialized weights.
pub fn gru_layer[T](ctx &autograd.Context[T], input_size int, hidden_size int) types.Layer[T] {
	unsafe {
		w_ih_shape := [3 * hidden_size, input_size]
		w_hh_shape := [3 * hidden_size, hidden_size]
		b_shape := [3 * hidden_size]
		w_ih := ctx.variable(internal.kaiming_normal[T](w_ih_shape))
		w_hh := ctx.variable(internal.kaiming_normal[T](w_hh_shape))
		b_ih := ctx.variable(vtl.zeros[T](b_shape))
		b_hh := ctx.variable(vtl.zeros[T](b_shape))

		mut layer := &GRULayer[T]{
			w_ih:        w_ih
			w_hh:        w_hh
			b_ih:        b_ih
			b_hh:        b_hh
			ctx:         ctx
			input_size:  input_size
			hidden_size: hidden_size
		}
		return types.Layer[T](layer)
	}
}

fn (l &GRULayer[T]) output_shape() []int {
	return [l.hidden_size]
}

fn (l &GRULayer[T]) variables() []&autograd.Variable[T] {
	return [l.w_ih, l.w_hh, l.b_ih, l.b_hh]
}

// forward runs the GRU over the input sequence.
// Input shape: [batch, seq_len, input_size]
// Internally transposes to [seq_len, batch, input_size] for the step loop.
// Returns the final hidden state: [batch, hidden_size]
fn (l &GRULayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	unsafe {
		batch := input.value.shape[0]
		// Transpose from [batch, seq_len, input_size] to [seq_len, batch, input_size]
		input_t := input.value.transpose([1, 0, 2])!

		hidden0 := vtl.zeros[T]([batch, l.hidden_size])

		_, h_n := internal.gru_forward_single[T](input_t, hidden0, l.w_ih.value, l.w_hh.value,
			l.b_ih.value, l.b_hh.value)!

		return l.ctx.variable(h_n)
	}
}
