module stats

import vtl
import math

fn test_quantiles_linear_sorts_once_and_interpolates() ! {
	values := vtl.from_1d([30.0, 0.0, 20.0, 10.0])!
	result := quantiles_linear(values, [0.0, 0.25, 0.5, 0.75, 1.0])!
	assert result.to_array() == [0.0, 7.5, 15.0, 22.5, 30.0]
	assert values.to_array() == [30.0, 0.0, 20.0, 10.0]
	integers := vtl.from_1d([1, 2, 3, 4])!
	assert quantiles_linear(integers, [0.5])!.to_array() == [2.5]
}

fn test_quantiles_propagate_or_ignore_nan_as_requested() ! {
	values := vtl.from_1d([1.0, math.nan(), 3.0])!
	ordinary := quantiles_linear(values, [0.0, 0.5, 1.0])!
	assert ordinary.to_array().all(math.is_nan(it))
	ignored := nanquantiles_linear(values, [0.0, 0.5, 1.0])!
	assert ignored.to_array() == [1.0, 2.0, 3.0]
	all_nan := vtl.from_1d([math.nan(), math.nan()])!
	assert nanquantiles_linear(all_nan, [0.25, 0.75])!.to_array().all(math.is_nan(it))
}

fn test_quantiles_reject_empty_inputs_and_invalid_levels() ! {
	values := vtl.from_1d([1.0, 2.0])!
	if _ := quantiles_linear(values, [0.5, 1.1]) {
		assert false, 'out-of-range quantile must return an error'
	} else {
		assert true
	}
	empty := vtl.from_1d([]f64{})!
	if _ := nanquantiles_linear(empty, [0.5]) {
		assert false, 'empty input must return an error'
	} else {
		assert true
	}
}

fn test_quantiles_axis_prepends_quantile_dimension_and_sorts_once() ! {
	values := vtl.from_array([9.0, 1.0, 8.0, 2.0, 7.0, 3.0], [2, 3])!
	result := quantiles_axis(values, [0.0, 0.5, 1.0], 1)!
	assert result.shape == [3, 2]
	assert result.to_array() == [1.0, 2.0, 8.0, 3.0, 9.0, 7.0]
	negative_axis := quantiles_axis(values, [0.5], -2)!
	assert negative_axis.shape == [1, 3]
	assert negative_axis.to_array() == [5.5, 4.0, 5.5]
}

fn test_nanquantiles_axis_ignores_nan_and_ordinary_variant_propagates() ! {
	values := vtl.from_array([1.0, math.nan(), 3.0, 5.0, math.nan(), 7.0], [3, 2])!
	ignored := nanquantiles_axis(values, [0.0, 0.5, 1.0], 0)!
	assert ignored.shape == [3, 2]
	assert ignored.to_array() == [1.0, 5.0, 2.0, 6.0, 3.0, 7.0]
	propagated := quantiles_axis(values, [0.25, 0.75], 0)!
	assert math.is_nan(propagated.to_array()[0])
	assert math.is_nan(propagated.to_array()[1])
	assert math.is_nan(propagated.to_array()[2])
	assert math.is_nan(propagated.to_array()[3])
	all_nan := vtl.from_1d([math.nan(), math.nan()])!
	assert nanquantiles_axis(all_nan, [0.25, 0.75], 0)!.to_array().all(math.is_nan(it))
}
