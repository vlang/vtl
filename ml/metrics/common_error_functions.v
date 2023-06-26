module metrics

import math
import vtl
import vtl.stats

[inline]
pub fn squared_error[T](y &vtl.Tensor[T], y_true &vtl.Tensor[T]) !&vtl.Tensor[T] {
	diff := y.subtract(y_true)!
	return diff.multiply(diff)
}

[inline]
pub fn mean_squared_error[T](y &vtl.Tensor[T], y_true &vtl.Tensor[T]) !T {
	return stats.mean[T](squared_error[T](y, y_true)!)
}

[inline]
pub fn relative_error[T](y &vtl.Tensor[T], y_true &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return y.nmap([y_true], fn [T](vals []T, i []int) T {
		denom := math.max(math.abs(vals[1]), math.abs(vals[0]))
		if denom == vtl.cast[T](0) {
			return vtl.cast[T](0)
		}
		return math.abs(math.abs(vals[1]) - math.abs(vals[0])) / denom
	})
}

[inline]
pub fn mean_relative_error[T](y &vtl.Tensor[T], y_true &vtl.Tensor[T]) !T {
	return stats.mean[T](relative_error[T](y, y_true)!)
}

[inline]
pub fn absolute_error[T](y &vtl.Tensor[T], y_true &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return y_true.subtract(y)!.abs()
}

[inline]
pub fn mean_absolute_error[T](y &vtl.Tensor[T], y_true &vtl.Tensor[T]) !T {
	return stats.mean[T](absolute_error[T](y, y_true)!)
}
