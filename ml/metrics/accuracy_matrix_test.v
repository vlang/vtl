module metrics

import vtl

pub fn test_accuracy_score() {
	y_pred := vtl.from_1d<int>([0, 2, 1, 3]) or { panic(@FN + ' failed') }
	y_true := vtl.from_1d<int>([0, 1, 2, 3]) or { panic(@FN + ' failed') }
	assert accuracy_score<int>(y_pred, y_true) or { panic(@FN + ' failed') } == 0.5
}
