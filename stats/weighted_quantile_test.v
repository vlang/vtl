module stats

import vtl

fn test_weighted_quantiles_match_inverted_cdf_semantics() ! {
	values := vtl.from_1d([1.0, 2.0, 3.0, 4.0])!
	weights := vtl.from_1d([1.0, 1.0, 1.0, 7.0])!
	assert quantile_weighted(values, weights, 0.5)! == 4.0
	assert quantiles_weighted(values, weights, [0.0, 0.25, 0.5, 1.0])!.to_array() == [
		1.0,
		3.0,
		4.0,
		4.0,
	]
	zero_weight := vtl.from_1d([0.0, 1.0, 0.0, 1.0])!
	assert quantile_weighted(values, zero_weight, 0.0)! == 2.0
	assert quantile_weighted(values, zero_weight, 0.5)! == 2.0
}

fn test_weighted_quantiles_validate_shape_and_weights() ! {
	values := vtl.from_1d([1.0, 2.0])!
	if _ := quantile_weighted(values, vtl.from_1d([1.0])!, 0.5) {
		assert false, 'mismatched weight shapes must fail'
	}
	if _ := quantile_weighted(values, vtl.from_1d([-1.0, 2.0])!, 0.5) {
		assert false, 'negative weights must fail'
	}
	if _ := quantile_weighted(values, vtl.from_1d([0.0, 0.0])!, 0.5) {
		assert false, 'zero total weight must fail'
	}
	assert quantile_weighted(values, vtl.from_1d([1.0, 1.0])!, 0.5)! == 1.0
}

fn test_weighted_quantiles_reduce_axis_with_shared_or_full_weights() ! {
	values := vtl.from_array([10.0, 7.0, 4.0, 3.0, 2.0, 1.0], [2, 3])!
	full_weights := vtl.from_array([1.0, 1.0, 7.0, 1.0, 1.0, 1.0], [2, 3])!
	by_column := quantile_weighted_axis(values, full_weights, 0.5, 0, false)!
	assert by_column.shape == [3]
	assert by_column.to_array() == [3.0, 2.0, 4.0]
	by_row := quantiles_weighted_axis(values, vtl.from_1d([1.0, 1.0, 7.0])!, [0.0, 0.5, 1.0], -1)!
	assert by_row.shape == [3, 2]
	assert by_row.to_array() == [4.0, 1.0, 4.0, 1.0, 10.0, 3.0]
	kept := quantile_weighted_axis(values, vtl.from_1d([1.0, 1.0, 7.0])!, 0.5, 1, true)!
	assert kept.shape == [2, 1]
	assert kept.to_array() == [4.0, 1.0]
}
