module metrics

import vtl

pub fn test_accuracy_score() {
	y_pred := vtl.from_1d([0, 2, 1, 3])!
	y_true := vtl.from_1d([0, 1, 2, 3])!
	assert accuracy_score(y_pred, y_true)! == 0.5
}

pub fn test_mean_absolute_error() {
	y_true := vtl.from_1d([0.9, 0.2, 0.1, 0.4, 0.9])!
	y := vtl.from_1d([1.0, 0.0, 0.0, 1.0, 1.0])!
	assert mean_absolute_error(y, y_true)! == (1.1 / 5.0)
}

pub fn test_mean_relative_error() {
	y_true := vtl.from_1d([0.0, 0.0, -1.0, 1e-8, 1e-8])!
	y := vtl.from_1d([0.0, -1.0, 0.0, 0.0, 1e-7])!
	result := relative_error(y, y_true)!
	expected := vtl.from_1d([0.0, 1.0, 1.0, 1.0, 0.9])!
	assert result.array_equal(expected)
	assert mean_relative_error(y, y_true)! == 0.78
}

pub fn test_mean_squared_error() {
	y_true := vtl.from_1d([0.9, 0.2, 0.1, 0.4, 0.9])!
	y := vtl.from_1d([1.0, 0.0, 0.0, 1.0, 1.0])!
	assert mean_squared_error(y, y_true)! == (0.43 / 5.0)
}
