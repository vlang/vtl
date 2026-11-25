// Package array_api is VTL's explicit, incrementally implemented Python Array
// API namespace. Operations are added only after their Array API semantics
// have focused regression coverage.
module array_api

import vtl
import vtl.la
import vtl.stats

// standard_version identifies the Array API specification targeted by this
// namespace. It is intentionally separate from VTL's package version.
pub const standard_version = '2024.12'

// Capability describes one operation exposed by this namespace.
pub struct Capability {
pub:
	operation string
	supported bool
}

// capability_matrix returns the current, deliberately small conformance
// surface. Consumers can use this instead of guessing from VTL's broader API.
pub fn capability_matrix() []Capability {
	return [
		Capability{'empty', true},
		Capability{'zeros', true},
		Capability{'ones', true},
		Capability{'full', true},
		Capability{'arange', true},
		Capability{'reshape', true},
		Capability{'broadcast_to', true},
		Capability{'transpose', true},
		Capability{'where', true},
		Capability{'matmul', true},
		Capability{'clip', true},
		Capability{'sum', true},
		Capability{'prod', true},
		Capability{'min', true},
		Capability{'max', true},
		Capability{'all', true},
		Capability{'any', true},
		Capability{'mean', true},
		Capability{'isclose', true},
		Capability{'allclose', true},
		Capability{'isfinite', true},
		Capability{'isinf', true},
		Capability{'isnan', true},
	]
}

// supports reports whether an operation is implemented in this namespace.
pub fn supports(operation string) bool {
	for capability in capability_matrix() {
		if capability.operation == operation {
			return capability.supported
		}
	}
	return false
}

// empty creates an array with the requested shape and element type.
pub fn empty[T](shape []int) &vtl.Tensor[T] {
	return vtl.empty[T](shape)
}

// zeros creates an array filled with zero.
pub fn zeros[T](shape []int) &vtl.Tensor[T] {
	return vtl.zeros[T](shape)
}

// ones creates an array filled with one.
pub fn ones[T](shape []int) &vtl.Tensor[T] {
	return vtl.ones[T](shape)
}

// full creates an array filled with value.
pub fn full[T](shape []int, value T) &vtl.Tensor[T] {
	return vtl.full[T](shape, value)
}

// arange creates values in the half-open interval [start, stop).
pub fn arange[T](start f64, stop f64, step f64) !&vtl.Tensor[T] {
	return vtl.arange[T](start, stop, step)
}

// reshape returns an array with the requested shape, preserving its values.
pub fn reshape[T](array &vtl.Tensor[T], shape []int) !&vtl.Tensor[T] {
	return array.reshape(shape)
}

// broadcast_to expands an array to a compatible shape.
pub fn broadcast_to[T](array &vtl.Tensor[T], shape []int) !&vtl.Tensor[T] {
	return array.broadcast_to(shape)
}

// transpose permutes axes, reversing them when no explicit order is supplied.
pub fn transpose[T](array &vtl.Tensor[T], axes ...int) !&vtl.Tensor[T] {
	mut order := axes.clone()
	if order.len == 0 {
		order = []int{len: array.rank(), init: array.rank() - index - 1}
	}
	return array.transpose(order)
}

// where selects values from x or y according to a broadcastable condition.
pub fn where[T](condition &vtl.Tensor[bool], x &vtl.Tensor[T], y &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return vtl.where(condition, x, y)
}

// matmul computes matrix multiplication with Array API vector and batch rules.
pub fn matmul[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return la.matmul(a, b)
}

// clip limits every array value to the inclusive scalar interval.
pub fn clip[T](array &vtl.Tensor[T], min_value T, max_value T) !&vtl.Tensor[T] {
	return array.clip(min_value, max_value)
}

// sum reduces all elements to a scalar in the input dtype.
pub fn sum[T vtl.Real](array &vtl.Tensor[T]) T {
	return stats.sum(array)
}

// prod reduces all elements to a product in the input dtype.
pub fn prod[T vtl.Real](array &vtl.Tensor[T]) T {
	return stats.prod(array)
}

// min reduces all elements to their minimum in the input dtype.
pub fn min[T vtl.Real](array &vtl.Tensor[T]) T {
	return stats.min(array)
}

// max reduces all elements to their maximum in the input dtype.
pub fn max[T vtl.Real](array &vtl.Tensor[T]) T {
	return stats.max(array)
}

// all reports whether every element is truthy.
pub fn all[T](array &vtl.Tensor[T]) bool {
	return array.all()
}

// any_ reports whether at least one element is truthy. The suffix is required
// because `any` is a reserved V type name.
pub fn any_[T](array &vtl.Tensor[T]) bool {
	return array.any()
}

// mean reduces real numeric elements to an f64 scalar.
pub fn mean[T vtl.Real](array &vtl.Tensor[T]) f64 {
	return stats.mean_as[T, f64](array)
}

// isclose compares broadcasted values using relative and absolute tolerances.
pub fn isclose[T](a &vtl.Tensor[T], b &vtl.Tensor[T], rtol f64, atol f64, equal_nan bool) !&vtl.Tensor[bool] {
	return a.isclose(b, rtol: rtol, atol: atol, equal_nan: equal_nan)
}

// allclose reports whether all broadcasted values satisfy the tolerances.
pub fn allclose[T](a &vtl.Tensor[T], b &vtl.Tensor[T], rtol f64, atol f64, equal_nan bool) !bool {
	return a.allclose(b, rtol: rtol, atol: atol, equal_nan: equal_nan)
}

// isfinite returns true for values that are neither infinite nor NaN.
pub fn isfinite[T](array &vtl.Tensor[T]) &vtl.Tensor[bool] {
	return array.is_finite()
}

// isinf returns true for positive or negative infinite values.
pub fn isinf[T](array &vtl.Tensor[T]) &vtl.Tensor[bool] {
	return array.is_inf(0)
}

// isnan returns true for NaN values.
pub fn isnan[T](array &vtl.Tensor[T]) &vtl.Tensor[bool] {
	return array.is_nan()
}
