module stats

import math
import vtl

fn test_nanmean_nanvar_nanstd_ignore_nan_and_use_ddof() ! {
	values := vtl.from_1d([1.0, math.nan(), 3.0, 5.0])!
	assert nanmean(values) == 3.0
	assert math.abs(nanvar(values, 0)! - 8.0 / 3.0) < 1e-12
	assert math.abs(nanvar(values, 1)! - 4.0) < 1e-12
	assert math.abs(nanstd(values, 1)! - 2.0) < 1e-12
}

fn test_nan_reductions_axis_keep_dimension_and_nan_only_slices() ! {
	values := vtl.from_array([1.0, math.nan(), 3.0, 5.0, math.nan(), math.nan()], [3, 2])!
	means := nanmean_axis(values, 0)!
	assert means.shape == [1, 2]
	assert means.get([0, 0]) == 2.0
	assert means.get([0, 1]) == 5.0
	row_means := nanmean_axis(values, -1)!
	assert row_means.shape == [3, 1]
	assert row_means.get([0, 0]) == 1.0
	assert row_means.get([1, 0]) == 4.0
	assert math.is_nan(row_means.get([2, 0]))
	variances := nanvar_axis(values, 1, 1)!
	assert math.is_nan(variances.get([0, 0]))
	assert variances.get([1, 0]) == 2.0
	assert math.is_nan(variances.get([2, 0]))
	deviations := nanstd_axis(values, 1, 1)!
	assert math.is_nan(deviations.get([0, 0]))
	assert math.abs(deviations.get([1, 0]) - math.sqrt(2.0)) < 1e-12
	assert math.is_nan(deviations.get([2, 0]))
}

fn test_nan_reductions_all_nan_and_invalid_ddof_or_axis() ! {
	values := vtl.from_1d([math.nan(), math.nan()])!
	assert math.is_nan(nanmean(values))
	assert math.is_nan(nanvar(values, 0)!)
	assert math.is_nan(nanstd(values, 0)!)
	if _ := nanvar(values, -1) {
		assert false, 'nanvar must reject negative ddof'
	} else {
		assert true
	}
	if _ := nanmean_axis(values, 1) {
		assert false, 'nanmean_axis must reject out-of-range axes'
	} else {
		assert true
	}
}

fn test_standard_statistical_reductions_along_axis_propagate_nan() ! {
	values := vtl.from_array([1.0, 10.0, 3.0, 20.0, 5.0, math.nan()], [3, 2])!
	means := mean_axis(values, 0)!
	assert means.shape == [1, 2]
	assert means.get([0, 0]) == 3.0
	assert math.is_nan(means.get([0, 1]))
	population := variance_axis(values, 1, 0)!
	assert population.shape == [3, 1]
	assert population.get([0, 0]) == 20.25
	assert population.get([1, 0]) == 72.25
	assert math.is_nan(population.get([2, 0]))
	sample_std := std_axis(values, 1, 1)!
	assert sample_std.get([0, 0]) == math.sqrt(40.5)
	assert math.is_nan(sample_std.get([2, 0]))
	if _ := variance_axis(values, 0, -1) {
		assert false, 'variance_axis must reject negative ddof'
	} else {
		assert true
	}
}

fn test_nan_quantiles_ignore_nan_globally_and_by_axis() ! {
	values := vtl.from_array([1.0, math.nan(), 3.0, 5.0, math.nan(), math.nan()], [3, 2])!
	assert nanquantile_linear(values, 0.5)! == 3.0
	assert nanpercentile_linear(values, 50)! == 3.0
	columns := nanquantile_axis(values, 0.5, 0)!
	assert columns.shape == [1, 2]
	assert columns.get([0, 0]) == 2.0
	assert columns.get([0, 1]) == 5.0
	rows := nanquantile_axis(values, 0.5, -1)!
	assert rows.shape == [3, 1]
	assert rows.get([0, 0]) == 1.0
	assert rows.get([1, 0]) == 4.0
	assert math.is_nan(rows.get([2, 0]))
}

fn test_nan_quantiles_validate_quantile_and_empty_input() ! {
	values := vtl.from_1d([math.nan(), math.nan()])!
	assert math.is_nan(nanquantile_linear(values, 0.5)!)
	assert math.is_nan(nanquantile_axis(values, 0.5, 0)!.get([0]))
	if _ := nanquantile_linear(values, 1.1) {
		assert false, 'out-of-range quantile must return an error'
	} else {
		assert true
	}
	empty := vtl.from_1d([]f64{})!
	if _ := nanquantile_linear(empty, 0.5) {
		assert false, 'empty input must return an error'
	} else {
		assert true
	}
}
