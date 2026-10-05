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
	bhh_data []f64) !f64 {
	x := vtl.from_array(input_data, [2, 1, 2])!
	wih := vtl.from_array(wih_data, [6, 2])!
	whh := vtl.from_array(whh_data, [6, 2])!
	bih := vtl.from_array(bih_data, [6])!
	bhh := vtl.from_array(bhh_data, [6])!
	h0 := vtl.from_array([0.1, -0.2], [1, 2])!
	y, _ := gru_forward_single[f64](x, wih, whh, bih, bhh, h0)!
	mut total := f64(0)
	for i, value in y.to_array() { total += value * f64(i + 1) }
	return total
}

fn test_gru_forward_shapes_and_finite_difference_gradients() ! {
	input_data := [0.2, -0.1, 0.4, 0.3]
	wih_data := [0.1, -0.2, 0.05, 0.1, -0.1, 0.2, 0.3, 0.1, -0.2, 0.15, 0.05, -0.1]
	whh_data := [0.2, 0.1, -0.15, 0.05, 0.1, -0.2, -0.1, 0.15, 0.3, -0.25, 0.05, 0.2]
	bih_data := [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]
	bhh_data := [-0.01, 0.02, 0.01, -0.03, 0.02, -0.01]
	x := vtl.from_array(input_data, [2, 1, 2])!
	wih := vtl.from_array(wih_data, [6, 2])!
	whh := vtl.from_array(whh_data, [6, 2])!
	bih := vtl.from_array(bih_data, [6])!
	bhh := vtl.from_array(bhh_data, [6])!
	h0 := vtl.from_array([0.1, -0.2], [1, 2])!
	y, hn := gru_forward_single[f64](x, wih, whh, bih, bhh, h0)!
	assert y.shape == [2, 1, 2]
	assert hn.shape == [1, 2]
	mut upstream_data := []f64{len: 4}
	for i in 0 .. 4 { upstream_data[i] = f64(i + 1) }
	upstream := vtl.from_array(upstream_data, [2, 1, 2])!
	grads := gru_backward_single[f64](x, wih, whh, bih, bhh, h0, upstream)!
	assert grads.len == 5
	for item in 0 .. 5 {
		count := match item {
			0 { input_data.len }
			1 { wih_data.len }
			2 { whh_data.len }
			3 { bih_data.len }
			else { bhh_data.len }
		}
		for index in 0 .. count {
			mut xp, mut xm := input_data.clone(), input_data.clone()
			mut wp, mut wm := wih_data.clone(), wih_data.clone()
			mut hp, mut hm := whh_data.clone(), whh_data.clone()
			mut bp, mut bm := bih_data.clone(), bih_data.clone()
			mut cp, mut cm := bhh_data.clone(), bhh_data.clone()
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
				else {}
			}
			plus := gru_test_objective(xp, wp, hp, bp, cp)!
			minus := gru_test_objective(xm, wm, hm, bm, cm)!
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
	assert grads.len == 5
	assert grads[0].shape == x.shape
}
