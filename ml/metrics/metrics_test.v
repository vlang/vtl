module metrics

import vtl

pub fn test_accuracy_score() {
	y_pred := vtl.from_1d([0, 2, 1, 3]) or { panic(@FN + ' failed') }
	y_true := vtl.from_1d([0, 1, 2, 3]) or { panic(@FN + ' failed') }
	assert accuracy_score(y_pred, y_true) or { panic(@FN + ' failed') } == 0.5
}

pub fn test_mean_absolute_error() {
	y_true := vtl.from_1d([0.9, 0.2, 0.1, 0.4, 0.9]) or { panic(@FN + ' failed') }
	y := vtl.from_1d([1.0, 0.0, 0.0, 1.0, 1.0]) or { panic(@FN + ' failed') }
	assert mean_absolute_error(y, y_true) or { panic(@FN + ' failed') } == (1.1 / 5.0)
}

pub fn test_mean_relative_error() {
	y_true := vtl.from_1d([0.0, 0.0, -1.0, 1e-8, 1e-8]) or { panic(@FN + ' failed') }
	y := vtl.from_1d([0.0, -1.0, 0.0, 0.0, 1e-7]) or { panic(@FN + ' failed') }
	result := relative_error(y, y_true) or { panic(@FN + ' failed') }
	expected := vtl.from_1d([0.0, 1.0, 1.0, 1.0, 0.9]) or { panic(@FN + ' failed') }
	assert result.array_equal(expected)
	assert mean_relative_error(y, y_true) or { panic(@FN + ' failed') } == 0.78
}

pub fn test_mean_squared_error() {
	y_true := vtl.from_1d([0.9, 0.2, 0.1, 0.4, 0.9]) or { panic(@FN + ' failed') }
	y := vtl.from_1d([1.0, 0.0, 0.0, 1.0, 1.0]) or { panic(@FN + ' failed') }
	assert mean_squared_error(y, y_true) or { panic(@FN + ' failed') } == (0.43 / 5.0)
}
