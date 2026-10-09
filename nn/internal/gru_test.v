module internal

import vtl

fn test_gru_forward_matches_reference_gate_equations() ! {
	x := vtl.from_array([0.5493061443340548, 0.5493061443340548], [2, 1, 1])!
	wih := vtl.from_array([0.0, 0.0, 1.0], [3, 1])!
	whh := vtl.zeros[f64]([3, 1])
	bias := vtl.zeros[f64]([3])
	h0 := vtl.zeros[f64]([1, 1])
	y, _ := gru_forward_single[f64](x, wih, whh, bias, bias, h0)!
	values := y.to_array()
	assert values[0] > 0.249999999 && values[0] < 0.250000001
	assert values[1] > 0.374999999 && values[1] < 0.375000001
}

fn gru_test_objective(input_data []f64, wih_data []f64, whh_data []f64, bih_data []f64,
	bhh_data []f64, h0_data []f64) !f64 {
	x := vtl.from_array(input_data, [2, 1, 2])!
	wih := vtl.from_array(wih_data, [6, 2])!
	whh := vtl.from_array(whh_data, [6, 2])!
	bih := vtl.from_array(bih_data, [6])!
	bhh := vtl.from_array(bhh_data, [6])!
	h0 := vtl.from_array(h0_data, [1, 2])!
	y, _ := gru_forward_single[f64](x, wih, whh, bih, bhh, h0)!
	mut total := f64(0)
	for i, value in y.to_array() { total += value * f64(i + 1) }
	return total
}

fn gru_test_objective_with_final_state(input_data []f64, wih_data []f64, whh_data []f64,
	bih_data []f64, bhh_data []f64, h0_data []f64) !f64 {
	x := vtl.from_array(input_data, [2, 1, 2])!
	wih := vtl.from_array(wih_data, [6, 2])!
	whh := vtl.from_array(whh_data, [6, 2])!
	bih := vtl.from_array(bih_data, [6])!
	bhh := vtl.from_array(bhh_data, [6])!
	h0 := vtl.from_array(h0_data, [1, 2])!
	y, final_state := gru_forward_single[f64](x, wih, whh, bih, bhh, h0)!
	mut total := f64(0)
	for i, value in y.to_array() { total += value * f64(i + 1) }
	final_values := final_state.to_array()
	return total + final_values[0] * 0.7 - final_values[1] * 0.4
}

fn test_gru_backward_propagates_final_state_gradient() ! {
	input_data := [0.2, -0.1, 0.4, 0.3]
	wih_data := [0.1, -0.2, 0.05, 0.1, -0.1, 0.2, 0.3, 0.1, -0.2, 0.15, 0.05, -0.1]
	whh_data := [0.2, 0.1, -0.15, 0.05, 0.1, -0.2, -0.1, 0.15, 0.3, -0.25, 0.05, 0.2]
	bih_data := [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]
	bhh_data := [-0.01, 0.02, 0.01, -0.03, 0.02, -0.01]
	h0_data := [0.1, -0.2]
	x := vtl.from_array(input_data, [2, 1, 2])!
	wih := vtl.from_array(wih_data, [6, 2])!
	whh := vtl.from_array(whh_data, [6, 2])!
	bih := vtl.from_array(bih_data, [6])!
	bhh := vtl.from_array(bhh_data, [6])!
	h0 := vtl.from_array(h0_data, [1, 2])!
	grad_output := vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 1, 2])!
	grad_final_state := vtl.from_array([0.7, -0.4], [1, 2])!
	grads := gru_backward_single_with_final_state[f64](x, wih, whh, bih, bhh, h0,
		grad_output, grad_final_state)!
	for index in 0 .. h0_data.len {
		mut plus_state := h0_data.clone()
		mut minus_state := h0_data.clone()
		eps := 1e-6
		plus_state[index] += eps
		minus_state[index] -= eps
		plus := gru_test_objective_with_final_state(input_data, wih_data, whh_data, bih_data,
			bhh_data, plus_state)!
		minus := gru_test_objective_with_final_state(input_data, wih_data, whh_data, bih_data,
			bhh_data, minus_state)!
		numerical := (plus - minus) / (2 * eps)
		analytic := grads[5].get_nth(index)
		delta := numerical - analytic
		assert delta > -1e-6 && delta < 1e-6, 'GRU final-state gradient for h0[${index}] mismatch: analytic ${analytic}, numerical ${numerical}'
	}
}

fn test_gru_forward_shapes_and_finite_difference_gradients() ! {
	input_data := [0.2, -0.1, 0.4, 0.3]
	wih_data := [0.1, -0.2, 0.05, 0.1, -0.1, 0.2, 0.3, 0.1, -0.2, 0.15, 0.05, -0.1]
	whh_data := [0.2, 0.1, -0.15, 0.05, 0.1, -0.2, -0.1, 0.15, 0.3, -0.25, 0.05, 0.2]
	bih_data := [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]
	bhh_data := [-0.01, 0.02, 0.01, -0.03, 0.02, -0.01]
	h0_data := [0.1, -0.2]
	x := vtl.from_array(input_data, [2, 1, 2])!
	wih := vtl.from_array(wih_data, [6, 2])!
	whh := vtl.from_array(whh_data, [6, 2])!
	bih := vtl.from_array(bih_data, [6])!
	bhh := vtl.from_array(bhh_data, [6])!
	h0 := vtl.from_array(h0_data, [1, 2])!
	y, hn := gru_forward_single[f64](x, wih, whh, bih, bhh, h0)!
	assert y.shape == [2, 1, 2]
	assert hn.shape == [1, 2]
	mut upstream_data := []f64{len: 4}
	for i in 0 .. 4 { upstream_data[i] = f64(i + 1) }
	upstream := vtl.from_array(upstream_data, [2, 1, 2])!
	grads := gru_backward_single[f64](x, wih, whh, bih, bhh, h0, upstream)!
	assert grads.len == 6
	for item in 0 .. 6 {
		count := match item {
			0 { input_data.len }
			1 { wih_data.len }
			2 { whh_data.len }
			3 { bih_data.len }
			4 { bhh_data.len }
			else { h0_data.len }
		}
		for index in 0 .. count {
			mut xp, mut xm := input_data.clone(), input_data.clone()
			mut wp, mut wm := wih_data.clone(), wih_data.clone()
			mut hp, mut hm := whh_data.clone(), whh_data.clone()
			mut bp, mut bm := bih_data.clone(), bih_data.clone()
			mut cp, mut cm := bhh_data.clone(), bhh_data.clone()
			mut sp, mut sm := h0_data.clone(), h0_data.clone()
			eps := 1e-6
			match item {
				0 {
					xp[index] += eps
					xm[index] -= eps
				}
				1 {
					wp[index] += eps
					wm[index] -= eps
				}
				2 {
					hp[index] += eps
					hm[index] -= eps
				}
				3 {
					bp[index] += eps
					bm[index] -= eps
				}
				4 {
					cp[index] += eps
					cm[index] -= eps
				}
				5 {
					sp[index] += eps
					sm[index] -= eps
				}
				else {}
			}
			plus := gru_test_objective(xp, wp, hp, bp, cp, sp)!
			minus := gru_test_objective(xm, wm, hm, bm, cm, sm)!
			numerical := (plus - minus) / (2 * eps)
			analytic := grads[item].get_nth(index)
			delta := numerical - analytic
			assert delta > -1e-6 && delta < 1e-6, 'GRU gradient ${item} mismatch: analytic ${analytic}, numerical ${numerical}'
		}
	}
}

fn test_gru_rejects_incompatible_dimensions() {
	x := vtl.zeros[f64]([1, 1, 2])
	wih := vtl.zeros[f64]([6, 2])
	whh := vtl.zeros[f64]([6, 2])
	b := vtl.zeros[f64]([6])
	h0 := vtl.zeros[f64]([2, 2])
	output, _ := gru_forward_single[f64](x, wih, whh, b, b, h0) or { return }
	assert output.shape == [1, 1, 2], 'expected invalid initial-state dimensions to fail'
}

fn test_gru_f32_forward_and_backward() ! {
	x := vtl.from_array([f32(0.2), -0.1], [1, 1, 2])!
	wih := vtl.from_array([f32(0.1), 0.2, -0.1, 0.1, 0.2, -0.2], [3, 2])!
	whh := vtl.from_array([f32(0.1), 0.2, -0.2], [3, 1])!
	bias := vtl.zeros[f32]([3])
	h0 := vtl.zeros[f32]([1, 1])
	y, hn := gru_forward_single[f32](x, wih, whh, bias, bias, h0)!
	assert y.shape == [1, 1, 1]
	assert hn.shape == [1, 1]
	grad := vtl.ones_like[f32](y)
	grads := gru_backward_single[f32](x, wih, whh, bias, bias, h0, grad)!
	assert grads.len == 6
	assert grads[0].shape == x.shape
}
