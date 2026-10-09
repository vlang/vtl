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

fn lstm_test_objective_with_final_state(input_values []f64, w_ih_values []f64,
	w_hh_values []f64, b_ih_values []f64, b_hh_values []f64, hidden_values []f64,
	cell_values []f64, grad_values []f64) !f64 {
	input := vtl.from_array(input_values, [2, 1, 1])!
	w_ih := vtl.from_array(w_ih_values, [4, 1])!
	w_hh := vtl.from_array(w_hh_values, [4, 1])!
	b_ih := vtl.from_array(b_ih_values, [4])!
	b_hh := vtl.from_array(b_hh_values, [4])!
	hidden := vtl.from_array(hidden_values, [1, 1])!
	cell := vtl.from_array(cell_values, [1, 1])!
	output, final_hidden, final_cell := lstm_forward_single_with_cell[f64](input, hidden, cell,
		w_ih, w_hh, b_ih, b_hh)!
	mut objective := f64(0)
	for i, value in output.to_array() { objective += value * grad_values[i] }
	return objective + 0.6 * final_hidden.get_nth(0) - 0.35 * final_cell.get_nth(0)
}

fn test_lstm_backward_propagates_final_hidden_and_cell_gradients() ! {
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
	grad_output := vtl.from_array(grad_values, [2, 1, 1])!
	grad_final_hidden := vtl.from_array([0.6], [1, 1])!
	grad_final_cell := vtl.from_array([-0.35], [1, 1])!
	grads := lstm_backward_single_with_final_state[f64](input, hidden, cell, w_ih, w_hh, b_ih,
		b_hh, grad_output, grad_final_hidden, grad_final_cell)!
	eps := 1e-6
	for item in 5 .. 7 {
		values := if item == 5 { hidden_values } else { cell_values }
		mut plus := values.clone()
		mut minus := values.clone()
		plus[0] += eps
		minus[0] -= eps
		mut plus_hidden := hidden_values.clone()
		mut minus_hidden := hidden_values.clone()
		mut plus_cell := cell_values.clone()
		mut minus_cell := cell_values.clone()
		if item == 5 {
			plus_hidden = plus.clone()
			minus_hidden = minus.clone()
		} else {
			plus_cell = plus.clone()
			minus_cell = minus.clone()
		}
		plus_objective := lstm_test_objective_with_final_state(input_values, w_ih_values,
			w_hh_values, b_ih_values, b_hh_values, plus_hidden, plus_cell, grad_values)!
		minus_objective := lstm_test_objective_with_final_state(input_values, w_ih_values,
			w_hh_values, b_ih_values, b_hh_values, minus_hidden, minus_cell, grad_values)!
		numerical := (plus_objective - minus_objective) / (2 * eps)
		analytic := grads[item].get_nth(0)
		delta := numerical - analytic
		assert delta > -1e-6 && delta < 1e-6, 'LSTM final-state gradient ${item} mismatch: analytic ${analytic}, numerical ${numerical}'
	}
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

fn test_lstm_forward_multi_uses_per_layer_initial_hidden_state() ! {
	input := vtl.zeros[f64]([1, 1, 1])
	hidden0 := vtl.from_array([0.7], [1, 1, 1])!
	w_ih := vtl.zeros[f64]([1, 4, 1])
	w_hh := vtl.from_array([0.0, 0.0, 1.0, 0.0], [1, 4, 1])!
	bias := vtl.zeros[f64]([1, 4])
	_, final_hidden := lstm_forward_multi[f64](input, hidden0, w_ih, w_hh, bias, bias)!
	assert final_hidden.get_nth(0) > 0.0, 'multi-layer LSTM must use the supplied initial hidden state'
}
