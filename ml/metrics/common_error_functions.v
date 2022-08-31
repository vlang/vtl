module metrics

import vtl
import vtl.stats

[inline]
pub fn squared_error<T>(y &vtl.Tensor<T>, y_true &vtl.Tensor<T>) &vtl.Tensor<T> {
	diff := y.substract(y_true)
	return diff.multiply(diff)
}

[inline]
pub fn mean_squared_error<T>(y &vtl.Tensor<T>, y_true &vtl.Tensor<T>) &vtl.Tensor<T> {
	return stats.mean(squared_error(y, y_true))
}

[inline]
pub fn absolute_error<T>(y &vtl.Tensor<T>, y_true &vtl.Tensor<T>) &vtl.Tensor<T> {
	return y_true.substract(y).abs()
}

[inline]
pub fn mean_absolute_error<T>(y &vtl.Tensor<T>, y_true &vtl.Tensor<T>) &vtl.Tensor<T> {
	return stats.mean(absolute_error(y, y_true))
}
