// Package array_api is VTL's explicit, incrementally implemented Python Array
// API namespace. Operations are added only after their Array API semantics
// have focused regression coverage.
module array_api

import vtl
import vtl.la

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
		Capability{'where', true},
		Capability{'matmul', true},
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

// where selects values from x or y according to a broadcastable condition.
pub fn where[T](condition &vtl.Tensor[bool], x &vtl.Tensor[T], y &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return vtl.where(condition, x, y)
}

// matmul computes matrix multiplication with Array API vector and batch rules.
pub fn matmul[T](a &vtl.Tensor[T], b &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return la.matmul(a, b)
}
