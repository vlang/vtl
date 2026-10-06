module la

import math
import vtl

fn test_covariance_and_correlation_match_numpy_rowvar_orientation() ! {
	data := vtl.from_2d([
		[f64(0), 2, 4],
		[1, 3, 5],
	])!
	sample_covariance := covariance_matrix[f64](data, true, 1)!
	assert sample_covariance.shape == [2, 2]
	assert sample_covariance.to_array() == [4.0, 4.0, 4.0, 4.0]
	population_covariance := covariance_matrix[f64](data, true, 0)!
	assert population_covariance.to_array() == [8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0, 8.0 / 3.0]
	correlation := correlation_matrix[f64](data, true)!
	assert correlation.to_array() == [1.0, 1.0, 1.0, 1.0]
}

fn test_covariance_and_correlation_support_columns_as_variables() ! {
	data := vtl.from_2d([
		[f64(1), 2],
		[3, 4],
		[5, 6],
	])!
	assert covariance_matrix[f64](data, false, 1)!.to_array() == [4.0, 4.0, 4.0, 4.0]
	assert correlation_matrix[f64](data, false)!.to_array() == [1.0, 1.0, 1.0, 1.0]
}

fn test_covariance_accepts_strided_integer_input() ! {
	data := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!.transpose([1, 0])!
	covariance := covariance_matrix[int](data, false, 1)!
	assert covariance.to_array() == [1.0, 1.0, 1.0, 1.0]
}

fn test_correlation_reports_nan_for_constant_variables() ! {
	data := vtl.from_2d([
		[f64(1), 1, 1],
		[2, 3, 4],
	])!
	correlation := correlation_matrix[f64](data, true)!
	assert math.is_nan(correlation.get([0, 0]))
	assert math.is_nan(correlation.get([0, 1]))
	assert math.is_nan(correlation.get([1, 0]))
	assert correlation.get([1, 1]) == 1
}

fn test_covariance_rejects_invalid_input() {
	one_dimensional := vtl.from_1d([1.0, 2.0, 3.0])!
	if _ := covariance_matrix[f64](one_dimensional, true, 0) {
		assert false, 'covariance_matrix must reject non-matrix input'
	}
	data := vtl.from_2d([[1.0]])!
	degenerate := covariance_matrix[f64](data, true, 1)!
	assert math.is_nan(degenerate.get([0, 0]))
	negative_ddof := covariance_matrix[f64](vtl.from_2d([[1.0, 3.0]])!, true, -1)!
	assert negative_ddof.get([0, 0]) == 2.0 / 3.0
	if _ := covariance_matrix[f64](vtl.from_array([]f64{}, [1, 0])!, true, 0) {
		assert false, 'covariance_matrix must reject data without observations'
	}
}
