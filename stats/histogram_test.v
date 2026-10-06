module stats

import math
import vtl

fn test_histogram_matches_numpy_rightmost_edge_rule() ! {
	data := vtl.from_1d([0, 1, 2, 3, 4])!
	result := histogram[int](data, 4)!
	assert result.counts.to_array() == [1, 1, 1, 2]
	assert result.bin_edges.to_array() == [0.0, 1.0, 2.0, 3.0, 4.0]
}

fn test_histogram_explicit_range_ignores_outside_and_nan() ! {
	data := vtl.from_1d([math.nan(), -1.0, 0, 1, 2, 3])!
	result := histogram_range[f64](data, 2, 0, 2)!
	assert result.counts.to_array() == [1, 2]
	assert result.bin_edges.to_array() == [0.0, 1.0, 2.0]
}

fn test_histogram_constant_and_empty_inputs() ! {
	constant := histogram(vtl.from_1d([2, 2])!, 4)!
	assert constant.counts.to_array() == [0, 0, 2, 0]
	assert constant.bin_edges.to_array() == [1.5, 1.75, 2.0, 2.25, 2.5]
	empty := histogram(vtl.from_array([]f64{}, [0])!, 2)!
	assert empty.counts.to_array() == [0, 0]
	assert empty.bin_edges.to_array() == [0.0, 0.5, 1.0]
}

fn test_histogram_rejects_invalid_bins_and_ranges() {
	data := vtl.from_1d([1.0, 2.0])!
	if _ := histogram[f64](data, 0) {
		assert false, 'histogram must reject a non-positive bin count'
	}
	if _ := histogram_range[f64](data, 2, 1, 1) {
		assert false, 'histogram must reject a degenerate range'
	}
	if _ := histogram(vtl.from_1d([math.inf(1)])!, 2) {
		assert false, 'automatic range must reject non-finite values'
	}
}
