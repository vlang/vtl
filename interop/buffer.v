module interop

import vtl
import vtl.storage

// BufferMetadata is the stable, ABI-friendly description of a buffer view.
pub struct BufferMetadata {
pub:
	shape        []int
	strides      []int
	offset       int
	element_size int
	mutable      bool
}

// BufferView describes a (possibly strided) view over a caller-owned CPU
// buffer. The view never clones the buffer; conversion to a tensor is either
// an explicit copy (to_tensor) or a guarded zero-copy wrap
// (as_borrowed_tensor).
pub struct BufferView[T] {
pub:
	data    []T
	shape   []int
	strides []int
	offset  int
	mutable bool
}

// borrow validates a strided view over `data`. Strides must be positive and
// non-overlapping so that every logical element maps to exactly one physical
// slot; overlapping layouts are rejected rather than silently aliased.
pub fn borrow[T](data []T, shape []int, strides []int, offset int, mutable bool) !BufferView[T] {
	if shape.len != strides.len {
		return error('shape and strides must have the same rank')
	}
	if offset < 0 {
		return error('offset must be non-negative')
	}
	for i, dim in shape {
		if dim < 0 {
			return error('shape dimensions must be non-negative')
		}
		if strides[i] <= 0 {
			return error('strides must be positive')
		}
	}
	mut seen := map[int]bool{}
	for n in 0 .. element_count(shape) {
		idx := offset + logical_offset(shape, strides, n)
		if idx >= data.len {
			return error('view index ${idx} is outside the buffer of ${data.len} elements')
		}
		if seen[idx] {
			return error('overlapping stride layout maps multiple logical elements to index ${idx}')
		}
		seen[idx] = true
	}
	return BufferView[T]{
		data:    data
		shape:   shape.clone()
		strides: strides.clone()
		offset:  offset
		mutable: mutable
	}
}

// contiguous describes a row-major contiguous view over the whole buffer.
pub fn contiguous[T](data []T, shape []int) !BufferView[T] {
	mut strides := []int{len: shape.len}
	mut acc := 1
	for i := shape.len - 1; i >= 0; i-- {
		strides[i] = acc
		acc *= shape[i]
	}
	return borrow[T](data, shape, strides, 0, true)
}

// element_count returns the number of logical elements in a shape.
pub fn element_count(shape []int) int {
	mut count := 1
	for dim in shape {
		if dim == 0 {
			return 0
		}
		count *= dim
	}
	return count
}

fn logical_offset(shape []int, strides []int, n int) int {
	mut remaining := n
	mut offset := 0
	for i := shape.len - 1; i >= 0; i-- {
		dim := shape[i]
		if dim == 0 {
			continue
		}
		offset += (remaining % dim) * strides[i]
		remaining /= dim
	}
	return offset
}

// metadata returns the ABI-friendly description of the view.
pub fn (v &BufferView[T]) metadata() BufferMetadata {
	return BufferMetadata{
		shape:        v.shape.clone()
		strides:      v.strides.clone()
		offset:       v.offset
		element_size: sizeof(T)
		mutable:      v.mutable
	}
}

// is_zero_copy_eligible reports whether the view can be wrapped without a
// copy: it must cover the entire buffer, start at offset zero, and be row-major
// contiguous.
pub fn (v &BufferView[T]) is_zero_copy_eligible() bool {
	if v.offset != 0 || v.data.len != element_count(v.shape) {
		return false
	}
	canonical := contiguous_strides(v.shape)
	for i, stride in v.strides {
		if stride != canonical[i] {
			return false
		}
	}
	return true
}

fn contiguous_strides(shape []int) []int {
	mut strides := []int{len: shape.len}
	mut acc := 1
	for i := shape.len - 1; i >= 0; i-- {
		strides[i] = acc
		acc *= shape[i]
	}
	return strides
}

// values gathers logical elements in row-major order. This is the copy
// fallback used by to_tensor and is always safe for strided views.
pub fn (v &BufferView[T]) values() []T {
	mut result := []T{len: element_count(v.shape)}
	for n in 0 .. result.len {
		result[n] = v.data[v.offset + logical_offset(v.shape, v.strides, n)]
	}
	return result
}

// to_tensor copies the logical elements into an owned tensor.
pub fn (v &BufferView[T]) to_tensor() !&vtl.Tensor[T] {
	return vtl.from_array[T](v.values(), v.shape)
}

// as_borrowed_tensor wraps eligible buffers without copying. The caller must
// keep the backing array alive and refrain from mutating it for the tensor's
// lifetime; call to_tensor otherwise.
pub fn (v &BufferView[T]) as_borrowed_tensor() !&vtl.Tensor[T] {
	if !v.is_zero_copy_eligible() {
		return error('view is not eligible for zero-copy; use to_tensor instead')
	}
	return &vtl.Tensor[T]{
		data:    storage.from_borrowed[T](v.data)
		memory:  .row_major
		size:    element_count(v.shape)
		shape:   v.shape.clone()
		strides: v.strides.clone()
	}
}
