module metrics

import vtl
import math.stats

// accuracy_score returns the proportion of correctly classified samples (as f32)
pub fn accuracy_score<T>(y_pred &vtl.Tensor<T>, y_true &vtl.Tensor<T>) ?f64 {
	equal := y_pred.equal(y_true)?
	equal_arr := equal.to_array<bool>().map(if it { 1.0 } else { 0.0 })
	return stats.mean<f64>(equal_arr)
}
