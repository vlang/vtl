module stats

import math
import vtl

fn test_histogram2d_uses_numpy_axis_order_and_includes_rightmost_edges() ! {
	x := vtl.from_1d([0.1, 0.1, 0.8, 1.0])!
	y := vtl.from_1d([0.2, 0.8, 0.2, 1.0])!
	result := histogram2d_edges[f64, f64](x, y, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0])!
	assert result.counts.shape == [2, 2]
	assert result.counts.to_array() == [1, 1, 1, 1]
	assert result.x_edges.to_array() == [0.0, 0.5, 1.0]
	assert result.y_edges.to_array() == [0.0, 0.5, 1.0]
}

fn test_histogram2d_fast_uniform_bins_respect_internal_edges() ! {
	x := vtl.from_1d([0.49, 0.5, 0.51])!
	y := vtl.from_1d([0.49, 0.5, 0.51])!
	uniform := histogram2d_edges[f64, f64](x, y, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0])!
	assert uniform.counts.to_array() == [1, 0, 0, 2]

	irregular := histogram2d_edges[f64, f64](x, y, [0.0, 0.1, 0.4, 1.0], [0.0, 0.1, 0.4, 1.0])!
	assert irregular.counts.to_array() == [0, 0, 0, 0, 0, 0, 0, 0, 3]
}

fn test_histogram2d_infers_ranges_and_handles_empty_or_constant_data() ! {
	x := vtl.from_1d([0, 1, 2, 3])!
	y := vtl.from_1d([0.0, 1, 0, 1])!
	result := histogram2d[int, f64](x, y, 2, 2)!
	assert result.counts.shape == [2, 2]
	mut total_count := 0
	for count in result.counts.to_array() {
		total_count += count
	}
	assert total_count == 4
	assert result.x_edges.to_array() == [0.0, 1.5, 3.0]

	empty := vtl.from_1d([]f64{})!
	empty_result := histogram2d[f64, f64](empty, empty, 2, 3)!
	assert empty_result.counts.to_array() == [0, 0, 0, 0, 0, 0]
	assert empty_result.x_edges.to_array() == [0.0, 0.5, 1.0]
	assert empty_result.y_edges.to_array() == [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]

	constant := vtl.from_1d([2, 2])!
	constant_result := histogram2d[int, int](constant, constant, 2, 1)!
	assert constant_result.x_edges.to_array() == [1.5, 2.0, 2.5]
	assert constant_result.counts.to_array() == [0, 2]
}

fn test_histogram2d_edges_ignore_outside_values_and_nan() ! {
	x := vtl.from_1d([0.5, 1.5, 3.0, math.nan()])!
	y := vtl.from_1d([0.5, 1.5, 2.5, 0.5])!
	result := histogram2d_edges[f64, f64](x, y, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])!
	assert result.counts.to_array() == [1, 0, 0, 1]
}

fn test_histogram2d_weighted_density_normalizes_by_each_bin_area() ! {
	x := vtl.from_1d([0.5, 0.5, 2.0, 3.0])!
	y := vtl.from_1d([1.0, 2.5, 1.5, 3.0])!
	weights := vtl.from_1d([1.0, 2, 3, 4])!
	x_edges := [0.0, 1.0, 3.0]
	y_edges := [0.0, 2.0, 3.0]
	weighted := histogram2d_weighted_edges[f64, f64, f64](x, y, weights, x_edges,
		y_edges, false)!
	assert weighted.counts.to_array() == [1.0, 2.0, 3.0, 4.0]

	density := histogram2d_weighted_edges[f64, f64, f64](x, y, weights, x_edges,
		y_edges, true)!
	assert math.abs(density.counts.get_nth(0) - 0.05) < 1e-12
	assert math.abs(density.counts.get_nth(1) - 0.2) < 1e-12
	assert math.abs(density.counts.get_nth(2) - 0.075) < 1e-12
	assert math.abs(density.counts.get_nth(3) - 0.2) < 1e-12
	mut integral := 0.0
	for x_bin in 0 .. 2 {
		for y_bin in 0 .. 2 {
			index := x_bin * 2 + y_bin
			area := (x_edges[x_bin + 1] - x_edges[x_bin]) * (y_edges[y_bin + 1] - y_edges[y_bin])
			integral += density.counts.get_nth(index) * area
		}
	}
	assert math.abs(integral - 1.0) < 1e-12
}

fn test_histogram2d_density_normalizes_in_range_samples() ! {
	x := vtl.from_1d([0.5, 0.5, 1.5, 9.0])!
	y := vtl.from_1d([0.5, 1.5, 0.5, 9.0])!
	density := histogram2d_density_edges[f64, f64](x, y, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])!
	assert density.counts.to_array() == [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]

	inferred := histogram2d_density[f64, f64](x, y, 2, 2)!
	mut integral := 0.0
	for x_bin in 0 .. 2 {
		for y_bin in 0 .. 2 {
			index := x_bin * 2 + y_bin
			x_width := inferred.x_edges.get_nth(x_bin + 1) - inferred.x_edges.get_nth(x_bin)
			y_width := inferred.y_edges.get_nth(y_bin + 1) - inferred.y_edges.get_nth(y_bin)
			area := x_width * y_width
			integral += inferred.counts.get_nth(index) * area
		}
	}
	assert math.abs(integral - 1.0) < 1e-12
}

fn test_histogram2d_rejects_invalid_shapes_bins_edges_and_weights() {
	x := vtl.from_1d([0.5, 1.0])!
	y := vtl.from_1d([0.5])!
	weights := vtl.from_1d([1.0, 1.0])!
	if _ := histogram2d[f64, f64](x, y, 2, 2) {
		assert false, 'histogram2d must reject inputs with different sizes'
	}
	if _ := histogram2d[f64, f64](x, vtl.from_1d([0.5, 1.0])!, 0, 2) {
		assert false, 'histogram2d must reject a non-positive bin count'
	}
	if _ := histogram2d_edges[f64, f64](x, vtl.from_1d([0.5, 1.0])!, [0.0, 1.0, 1.0], [
		0.0,
		1.0,
	]) {
		assert false, 'histogram2d must reject repeated edges'
	}
	if _ := histogram2d_weighted_edges[f64, f64, f64](x, vtl.from_1d([0.5, 1.0])!,
		vtl.from_1d([1.0])!, [0.0, 1.0], [0.0, 1.0], false) {
		assert false, 'histogram2d must reject weights with the wrong size'
	}
	if _ := histogram2d[f64, f64](vtl.from_array([0.5, 1.0], [1, 2])!,
		vtl.from_1d([0.5, 1.0])!, 2, 2) {
		assert false, 'histogram2d must reject non-vector inputs'
	}
	if _ := histogram2d_weighted_edges[f64, f64, f64](x, vtl.from_1d([0.5, 1.0])!,
		vtl.from_array([1.0, 1.0], [1, 2])!, [0.0, 1.0], [0.0, 1.0], false) {
		assert false, 'histogram2d must reject non-vector weights'
	}
	if _ := histogram2d_weighted_edges[f64, f64, f64](x, vtl.from_1d([0.5, 1.0])!,
		vtl.from_1d([0.0, 0.0])!, [0.0, 1.0], [0.0, 1.0], true) {
		assert false, 'histogram2d density must reject zero total weight'
	}
}
