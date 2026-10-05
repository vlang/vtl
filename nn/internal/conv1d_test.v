module internal

import vtl

fn conv1d_test_objective(x_data []f64, w_data []f64, b_data []f64) !f64 {
	x := vtl.from_array(x_data, [1, 1, 3])!
	w := vtl.from_array(w_data, [1, 1, 2])!
	b := vtl.from_array(b_data, [1])!
	y := conv1d_forward[f64](x, w, b, Conv1DConfig{ padding: 1 })!
	mut total := f64(0)
	for i, value in y.to_array() { total += value * [0.5, -0.25, 1.5, 0.75][i] }
	return total
}

fn test_conv1d_matches_reference_and_supports_stride_padding_dilation() ! {
	x := vtl.from_array([1.0, 2, 3, 4], [1, 1, 4])!
	w := vtl.from_array([1.0, 10], [1, 1, 2])!
	b := vtl.from_array([0.5], [1])!
	y := conv1d_forward[f64](x, w, b, Conv1DConfig{})!
	assert y.shape == [1, 1, 3]
	assert y.to_array() == [21.5, 32.5, 43.5]
	strided := conv1d_forward[f64](x, w, b, Conv1DConfig{ stride: 2, padding: 1 })!
	assert strided.shape == [1, 1, 3]
	assert strided.to_array() == [10.5, 32.5, 4.5]
	dilated := conv1d_forward[f64](x, w, b, Conv1DConfig{ dilation: 2 })!
	assert dilated.shape == [1, 1, 2]
	assert dilated.to_array() == [31.5, 42.5]
}

fn test_conv1d_backward_matches_finite_differences() ! {
	x_data := [0.2, -0.1, 0.4]
	w_data := [0.3, -0.2]
	b_data := [0.05]
	x := vtl.from_array(x_data, [1, 1, 3])!
	w := vtl.from_array(w_data, [1, 1, 2])!
	b := vtl.from_array(b_data, [1])!
	grad := vtl.from_array([0.5, -0.25, 1.5, 0.75], [1, 1, 4])!
	analytic := conv1d_backward[f64](grad, x, w, b, Conv1DConfig{ padding: 1 })!
	for tensor_index in 0 .. 3 {
		count := match tensor_index {
			0 { x_data.len }
			1 { w_data.len }
			else { b_data.len }
		}
		for index in 0 .. count {
			mut xp, mut xm := x_data.clone(), x_data.clone()
			mut wp, mut wm := w_data.clone(), w_data.clone()
			mut bp, mut bm := b_data.clone(), b_data.clone()
			eps := 1e-6
			match tensor_index {
				0 {
					xp[index] += eps
					xm[index] -= eps
				}
				1 {
					wp[index] += eps
					wm[index] -= eps
				}
				else {
					bp[index] += eps
					bm[index] -= eps
				}
			}
			plus := conv1d_test_objective(xp, wp, bp)!
			minus := conv1d_test_objective(xm, wm, bm)!
			delta := (plus - minus) / (2 * eps) - analytic[tensor_index].get_nth(index)
			assert delta > -1e-6 && delta < 1e-6, 'Conv1D gradient ${tensor_index}:${index} differs by ${delta}'
		}
	}
}

fn test_conv1d_grouped_channels() ! {
	x := vtl.from_array([1.0, 2, 3, 4], [1, 2, 2])!
	w := vtl.from_array([2.0, 3], [2, 1, 1])!
	b := vtl.zeros[f64]([2])
	y := conv1d_forward[f64](x, w, b, Conv1DConfig{ groups: 2 })!
	assert y.to_array() == [2.0, 4, 9, 12]
}
