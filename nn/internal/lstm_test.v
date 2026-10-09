module internal

import vtl

fn lstm_test_objective(input_values []f64, w_ih_values []f64, w_hh_values []f64,
	b_ih_values []f64, b_hh_values []f64, hidden_values []f64, cell_values []f64,
	grad_values []f64) !f64 {
	input := vtl.from_array(input_values, [2, 1, 1])!
	w_ih := vtl.from_array(w_ih_values, [4, 1])!
	w_hh := vtl.from_array(w_hh_values, [4, 1])!
	b_ih := vtl.from_array(b_ih_values, [4])!
	b_hh := vtl.from_array(b_hh_values, [4])!
	hidden := vtl.from_array(hidden_values, [1, 1])!
	cell := vtl.from_array(cell_values, [1, 1])!
	gradient := vtl.from_array(grad_values, [2, 1, 1])!
	output, _, _ := lstm_forward_single_with_cell[f64](input, hidden, cell, w_ih, w_hh, b_ih,
		b_hh)!
	mut objective := f64(0)
	for i, value in output.to_array() {
		objective += value * grad_values[i]
	}
	return objective
}

fn test_lstm_forward_shapes_and_bptt_gradients_match_finite_differences() ! {
	input_values := [0.2, -0.1]
	w_ih_values := [0.1, -0.2, 0.05, 0.1]
	w_hh_values := [0.2, -0.15, 0.1, -0.2]
	b_ih_values := [0.01, -0.02, 0.03, 0.01]
	b_hh_values := [-0.01, 0.02, -0.01, 0.03]
	hidden_values := [0.1]
	cell_values := [-0.2]
	grad_values := [0.7, -0.4]
	input := vtl.from_array(input_values, [2, 1, 1])!
	w_ih := vtl.from_array(w_ih_values, [4, 1])!
	w_hh := vtl.from_array(w_hh_values, [4, 1])!
	b_ih := vtl.from_array(b_ih_values, [4])!
	b_hh := vtl.from_array(b_hh_values, [4])!
	hidden := vtl.from_array(hidden_values, [1, 1])!
	cell := vtl.from_array(cell_values, [1, 1])!
	gradient := vtl.from_array(grad_values, [2, 1, 1])!
	output, final_hidden, final_cell := lstm_forward_single_with_cell[f64](input, hidden, cell,
		w_ih, w_hh, b_ih, b_hh)!
	assert output.shape == [2, 1, 1]
	assert final_hidden.shape == [1, 1]
	assert final_cell.shape == [1, 1]
	grads := lstm_backward_single[f64](input, hidden, cell, w_ih, w_hh, b_ih, b_hh, gradient)!
	assert grads.len == 7
	arrays := [input_values, w_ih_values, w_hh_values, b_ih_values, b_hh_values, hidden_values,
		cell_values]
	eps := 1e-6
	for item in 0 .. arrays.len {
		for index in 0 .. arrays[item].len {
			mut plus := arrays.clone()
			mut minus := arrays.clone()
			plus[item] = arrays[item].clone()
			minus[item] = arrays[item].clone()
			plus[item][index] += eps
			minus[item][index] -= eps
			plus_obj := lstm_test_objective(plus[0], plus[1], plus[2], plus[3], plus[4], plus[5],
				plus[6], grad_values)!
			minus_obj := lstm_test_objective(minus[0], minus[1], minus[2], minus[3], minus[4],
				minus[5], minus[6], grad_values)!
			numerical := (plus_obj - minus_obj) / (2 * eps)
			analytic := grads[item].get_nth(index)
			delta := numerical - analytic
			assert delta > -1e-6 && delta < 1e-6, 'LSTM gradient ${item} mismatch: analytic ${analytic}, numerical ${numerical}'
		}
	}
}

fn test_lstm_rejects_incompatible_dimensions() {
	input := vtl.zeros[f64]([1, 1, 2])
	hidden := vtl.zeros[f64]([1, 1])
	cell := vtl.zeros[f64]([1, 1])
	w_ih := vtl.zeros[f64]([3, 2])
	w_hh := vtl.zeros[f64]([4, 1])
	bias := vtl.zeros[f64]([4])
	if _, _, _ := lstm_forward_single_with_cell[f64](input, hidden, cell, w_ih, w_hh, bias,
		bias) {
		assert false, 'LSTM must reject a weight tensor without all four gate matrices'
	} else {
		assert true
	}
}
