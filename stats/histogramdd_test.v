module stats

import math
import vtl

fn test_histogramdd_edges_counts_nd_samples_and_includes_final_edges() ! {
	sample := vtl.from_array[f64]([0.1, 0.2, 0.2, 0.7, 0.6, 0.9, 1.0, 1.0, 1.0], [3, 3])!
	mut edges := [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]
	result := histogramdd_edges[f64](sample, edges)!

	assert result.counts.shape == [2, 2, 2]
	assert result.counts.to_array() == [1, 0, 0, 0, 0, 0, 0, 2]
	assert result.edges == edges
	edges[0][0] = -1.0
	assert result.edges[0][0] == 0.0
}

fn test_histogramdd_infers_axis_ranges_and_handles_empty_or_constant_data() ! {
	sample := vtl.from_array[int]([0, 10, 1, 10, 1, 20], [3, 2])!
	result := histogramdd[int](sample, [2, 5])!
	assert result.counts.shape == [2, 5]
	assert result.edges[0] == [0.0, 0.5, 1.0]
	assert result.edges[1] == [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
	assert result.counts.to_array() == [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
	weighted := histogramdd_weighted[int, f64](sample, vtl.from_1d([1.0, 2.0, 3.0])!, [
		1,
		2,
	],
		false)!
	assert weighted.counts.shape == [1, 2]
	assert weighted.counts.to_array() == [3.0, 3.0]

	empty := vtl.from_array[f64]([], [0, 2])!
	empty_result := histogramdd[f64](empty, [2, 1])!
	assert empty_result.counts.shape == [2, 1]
	assert empty_result.counts.to_array() == [0, 0]
	assert empty_result.edges == [[0.0, 0.5, 1.0], [0.0, 1.0]]

	constant := vtl.from_array[int]([4, 7, 4, 7], [2, 2])!
	constant_result := histogramdd[int](constant, [1, 1])!
	assert constant_result.edges == [[3.5, 4.5], [6.5, 7.5]]
	assert constant_result.counts.to_array() == [2]
}

fn test_histogramdd_skips_out_of_range_and_nan_points() ! {
	sample := vtl.from_array[f64]([0.5, 0.5, 1.5, 1.5, 2.1, 0.5, math.nan(), 0.5], [4, 2])!
	result := histogramdd_edges[f64](sample, [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])!
	assert result.counts.to_array() == [1, 0, 0, 1]
}

fn test_histogramdd_uses_nonuniform_edges_and_assigns_internal_edges_right() ! {
	sample := vtl.from_array[f64]([0.1, 1.0, 5.0, 0.2, 2.0, 10.0, 0.5, 4.0, 9.0], [3, 3])!
	edges := [[0.0, 0.2, 1.0], [0.0, 2.0, 5.0], [0.0, 10.0]]
	result := histogramdd_edges[f64](sample, edges)!
	assert result.counts.shape == [2, 2, 1]
	assert result.counts.to_array() == [1, 0, 0, 2]
}

fn test_histogramdd_weighted_density_normalizes_by_bin_volume() ! {
	sample := vtl.from_array[f64]([0.25, 0.5, 2.0, 1.5, 2.0, 3.0], [2, 3])!
	weights := vtl.from_1d([1.0, 3.0])!
	edges := [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 4.0]]
	result := histogramdd_weighted_edges[f64, f64](sample, weights, edges, true)!
	values := result.counts.to_array()
	assert values[0] == 1.0 / 24.0
	assert values[7] == 3.0 / 8.0
	assert math.abs(values[0] * 6.0 + values[7] * 2.0 - 1.0) < 1e-12
	assert result.edges == edges
}

fn test_histogramdd_density_normalizes_across_nonuniform_bin_volumes() ! {
	sample := vtl.from_array[f64]([0.25, 0.5, 2.0, 1.5, 2.0, 3.0], [2, 3])!
	edges := [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 4.0]]
	result := histogramdd_density_edges[f64](sample, edges)!
	values := result.counts.to_array()
	assert values[0] == 0.5 / 6.0
	assert values[7] == 0.5 / 2.0
	assert math.abs(values[0] * 6.0 + values[7] * 2.0 - 1.0) < 1e-12

	weighted_counts := histogramdd_weighted_edges[f64, f64](sample,
		vtl.from_1d([1.0, 3.0])!, edges, false)!
	assert weighted_counts.counts.to_array()[0] == 1.0
	assert weighted_counts.counts.to_array()[7] == 3.0
}

fn test_histogramdd_rejects_invalid_shapes_bins_edges_and_weights() ! {
	valid := vtl.from_array[f64]([0.0, 1.0, 2.0, 3.0], [2, 2])!
	if _ := histogramdd[f64](valid, [2]) {
		assert false, 'histogramdd must require a bin count for every dimension'
	}
	if _ := histogramdd[f64](valid, [2, 0]) {
		assert false, 'histogramdd must reject non-positive bin counts'
	}
	if _ := histogramdd_bin_count([max_int, 2]) {
		assert false, 'histogramdd must reject bin-shape multiplication overflow'
	}
	if _ := histogramdd_edges[f64](valid, [[0.0, 1.0]]) {
		assert false, 'histogramdd must require edges for every dimension'
	}
	if _ := histogramdd_edges[f64](valid, [[0.0, 1.0], [0.0, 1.0, 1.0]]) {
		assert false, 'histogramdd must reject repeated edges'
	}
	if _ := histogramdd_edges[f64](vtl.from_1d([1.0, 2.0])!, [[0.0, 1.0]]) {
		assert false, 'histogramdd must require a rank-two sample tensor'
	}
	if _ := histogramdd_weighted_edges[f64, f64](valid, vtl.from_1d([0.0, 0.0])!, [
		[0.0, 1.0],
		[0.0, 1.0],
	],
		true) {
		assert false, 'histogramdd density must reject zero total weight'
	}
	if _ := histogramdd_weighted_edges[f64, f64](valid, vtl.from_1d([1.0])!, [
		[0.0, 1.0],
		[0.0, 1.0],
	], false) {
		assert false, 'histogramdd must reject weights with the wrong size'
	}
}
