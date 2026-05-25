module main

import vtl
import vtl.ml.metrics

fn main() {
	// Predicted and ground-truth class indices
	y_pred := vtl.from_1d([0, 2, 1, 3]) or { panic(err) }
	y_true := vtl.from_1d([0, 1, 2, 3]) or { panic(err) }

	acc := metrics.accuracy_score(y_pred, y_true) or { panic(err) }
	println('accuracy: ${acc:.2f}')
}
