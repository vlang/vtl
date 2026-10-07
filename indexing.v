module vtl

// IndexKind describes one axis operation used by mixed_index.
pub enum IndexKind {
	full
	slice
	integer
	array
}

struct AxisRange {
	start  int
	length int
	step   int
}

// TensorIndex describes one axis of a mixed basic and advanced indexing
// operation. Use the constructors below to create valid values.
pub struct TensorIndex {
pub:
	kind      IndexKind
	integer   int
	start     int
	stop      int
	has_start bool
	has_stop  bool
	step      int
	indices   &Tensor[int]
}

// full_index selects every position on one axis.
pub fn full_index() TensorIndex {
	return TensorIndex{
		kind:    .full
		step:    1
		indices: unsafe { nil }
	}
}

// slice_index selects an explicit Python-style [start:stop:step] range.
// Negative boundaries count from the end; a zero step is rejected.
pub fn slice_index(start int, stop int, step int) !TensorIndex {
	if step == 0 {
		return error('slice_index: step cannot be zero')
	}
	return TensorIndex{
		kind:      .slice
		start:     start
		stop:      stop
		has_start: true
		has_stop:  true
		step:      step
		indices:   unsafe { nil }
	}
}

// slice_all selects the whole axis, optionally traversing it in reverse.
pub fn slice_all(step int) !TensorIndex {
	if step == 0 {
		return error('slice_index: step cannot be zero')
	}
	return TensorIndex{
		kind:    .slice
		step:    step
		indices: unsafe { nil }
	}
}

// slice_from selects from start through the end of the axis in step
// increments. A negative step traverses toward the beginning.
pub fn slice_from(start int, step int) !TensorIndex {
	if step == 0 {
		return error('slice_index: step cannot be zero')
	}
	return TensorIndex{
		kind:      .slice
		start:     start
		has_start: true
		step:      step
		indices:   unsafe { nil }
	}
}

// slice_to selects from the beginning of the axis through stop in step
// increments.
pub fn slice_to(stop int, step int) !TensorIndex {
	if step == 0 {
		return error('slice_index: step cannot be zero')
	}
	return TensorIndex{
		kind:     .slice
		stop:     stop
		has_stop: true
		step:     step
		indices:  unsafe { nil }
	}
}

// integer_index selects one position and removes its axis from the result.
pub fn integer_index(index int) TensorIndex {
	return TensorIndex{
		kind:    .integer
		integer: index
		indices: unsafe { nil }
	}
}

// array_index selects positions using an integer coordinate tensor. All
// coordinate tensors in a mixed_index call are broadcast together.
pub fn array_index(indices &Tensor[int]) TensorIndex {
	return TensorIndex{
		kind:    .array
		indices: indices
	}
}

// mixed_index combines full slices, ranges, scalar integers, and broadcasted
// coordinate tensors. Omitted trailing axes select the full axis. If every
// index is basic, the result is a view; if any coordinate tensor is used, the
// result is an independent row-major copy, matching NumPy's indexing rules.
pub fn mixed_index[T](t &Tensor[T], indices []TensorIndex) !&Tensor[T] {
	rank := t.rank()
	if indices.len > rank {
		return error('mixed_index: got ${indices.len} axis indices for rank ${rank}')
	}
	mut axis_indices := []TensorIndex{len: rank, init: full_index()}
	for axis, index in indices {
		axis_indices[axis] = index
	}

	mut ranges := []AxisRange{len: rank}
	mut slice_axes := []int{cap: rank}
	mut coordinate_axes := []int{cap: rank}
	mut first_advanced_axis := rank
	mut last_advanced_axis := -1
	for axis, index in axis_indices {
		match index.kind {
			.full {
				ranges[axis] = axis_range(t.shape[axis])
				slice_axes << axis
			}
			.slice {
				ranges[axis] = normalized_slice_range(t.shape[axis], index) or {
					return error('mixed_index: axis ${axis}: ${err}')
				}
				slice_axes << axis
			}
			.integer {
				normalized := normalize_scalar_index(index.integer, t.shape[axis], axis) or {
					return err
				}
				axis_indices[axis].integer = normalized
				first_advanced_axis = if axis < first_advanced_axis {
					axis
				} else {
					first_advanced_axis
				}
				last_advanced_axis = axis
			}
			.array {
				if isnil(index.indices) {
					return error('mixed_index: coordinate tensor for axis ${axis} is nil')
				}
				coordinate_axes << axis
				first_advanced_axis = if axis < first_advanced_axis {
					axis
				} else {
					first_advanced_axis
				}
				last_advanced_axis = axis
			}
		}
	}

	if coordinate_axes.len == 0 {
		return mixed_basic_index[T](t, axis_indices, ranges, slice_axes)
	}

	mut coordinate_tensors := []&Tensor[int]{cap: coordinate_axes.len}
	for axis in coordinate_axes {
		coordinate_tensors << axis_indices[axis].indices
	}
	broadcast_coordinates := broadcast_n[int](coordinate_tensors) or {
		return error('mixed_index: coordinate tensors cannot be broadcast: ${err}')
	}
	advanced_shape := broadcast_coordinates[0].shape
	advanced_rank := advanced_shape.len

	mut advanced_contiguous := true
	for axis in first_advanced_axis .. last_advanced_axis + 1 {
		if axis_indices[axis].kind == .full || axis_indices[axis].kind == .slice {
			advanced_contiguous = false
			break
		}
	}
	mut output_shape := []int{cap: rank + advanced_rank}
	if advanced_contiguous {
		for axis in slice_axes {
			if axis < first_advanced_axis {
				output_shape << ranges[axis].length
			}
		}
		output_shape << advanced_shape
		for axis in slice_axes {
			if axis > last_advanced_axis {
				output_shape << ranges[axis].length
			}
		}
	} else {
		output_shape << advanced_shape
		for axis in slice_axes {
			output_shape << ranges[axis].length
		}
	}

	mut result := empty[T](output_shape, memory: .row_major)
	mut output_index := []int{len: output_shape.len}
	mut advanced_index := []int{len: advanced_rank}
	mut input_index := []int{len: rank}
	mut slice_cursor := 0
	mut coordinate_cursor := 0
	for flat_index in 0 .. result.size {
		decode_flat_coordinate(flat_index, output_shape, mut output_index)
		slice_cursor = 0
		if advanced_contiguous {
			for axis in slice_axes {
				if axis < first_advanced_axis {
					input_index[axis] = ranges[axis].start + output_index[slice_cursor] * ranges[axis].step
					slice_cursor++
				}
			}
			for i in 0 .. advanced_rank {
				advanced_index[i] = output_index[slice_cursor + i]
			}
			slice_cursor += advanced_rank
			for axis in slice_axes {
				if axis > last_advanced_axis {
					input_index[axis] = ranges[axis].start + output_index[slice_cursor] * ranges[axis].step
					slice_cursor++
				}
			}
		} else {
			for i in 0 .. advanced_rank {
				advanced_index[i] = output_index[i]
			}
			slice_cursor = advanced_rank
			for axis in slice_axes {
				input_index[axis] = ranges[axis].start + output_index[slice_cursor] * ranges[axis].step
				slice_cursor++
			}
		}

		coordinate_cursor = 0
		for axis, index in axis_indices {
			match index.kind {
				.full, .slice {
					continue
				}
				.integer {
					input_index[axis] = index.integer
				}
				.array {
					selected := broadcast_coordinates[coordinate_cursor].get(advanced_index)
					input_index[axis] = normalize_scalar_index(selected, t.shape[axis], axis) or {
						return err
					}
					coordinate_cursor++
				}
			}
		}
		result.data.data[flat_index] = t.get(input_index)
	}
	return result
}

fn mixed_basic_index[T](t &Tensor[T], indices []TensorIndex, ranges []AxisRange, slice_axes []int) !&Tensor[T] {
	mut shape := []int{cap: slice_axes.len}
	mut strides := []int{cap: slice_axes.len}
	mut offset := 0
	mut has_negative_step := false
	for axis, index in indices {
		match index.kind {
			.full, .slice {
				range := ranges[axis]
				shape << range.length
				strides << t.strides[axis] * range.step
				if range.step < 0 {
					has_negative_step = true
				}
				if range.length > 0 {
					offset += t.strides[axis] * range.start
				}
			}
			.integer {
				offset += t.strides[axis] * index.integer
			}
			.array {
				return error('mixed_index: internal error: coordinate index in basic path')
			}
		}
	}
	if has_negative_step {
		mut result := empty[T](shape, memory: .row_major)
		mut output_index := []int{len: shape.len}
		mut input_index := []int{len: t.rank()}
		for flat_index in 0 .. result.size {
			decode_flat_coordinate(flat_index, shape, mut output_index)
			mut output_axis := 0
			for axis, index in indices {
				match index.kind {
					.full, .slice {
						range := ranges[axis]
						input_index[axis] = range.start + output_index[output_axis] * range.step
						output_axis++
					}
					.integer {
						input_index[axis] = index.integer
					}
					.array {
						return error('mixed_index: internal error: coordinate index in basic path')
					}
				}
			}
			result.data.data[flat_index] = t.get(input_index)
		}
		return result
	}
	mut result := &Tensor[T]{
		shape:   shape
		strides: strides
		size:    size_from_shape(shape)
		data:    t.data.offset[T](offset)
		memory:  .row_major
	}
	result.ensure_memory()
	return result
}

fn axis_range(length int) AxisRange {
	return AxisRange{
		length: length
		step:   1
	}
}

fn normalize_scalar_index(index int, length int, axis int) !int {
	if index < -length || index >= length {
		return error('index ${index} is out of range for axis ${axis} with size ${length}')
	}
	normalized := if index < 0 { index + length } else { index }
	return normalized
}

fn normalized_slice_range(length int, index TensorIndex) !AxisRange {
	if index.step == 0 {
		return error('slice step cannot be zero')
	}
	if index.step > 0 {
		mut first := if index.has_start { index.start } else { 0 }
		mut limit := if index.has_stop { index.stop } else { length }
		if index.has_start && first < 0 {
			first = if first < -length { 0 } else { first + length }
		}
		if index.has_stop && limit < 0 {
			limit = if limit < -length { 0 } else { limit + length }
		}
		first = clamp_index(first, 0, length)
		limit = clamp_index(limit, 0, length)
		distance := limit - first
		count := distance / index.step + if distance % index.step == 0 { 0 } else { 1 }
		return AxisRange{
			start:  first
			length: count
			step:   index.step
		}
	} else {
		mut first := if index.has_start { index.start } else { length - 1 }
		mut limit := if index.has_stop { index.stop } else { -1 }
		if index.has_start && first < 0 {
			first = if first < -length { -1 } else { first + length }
		}
		if index.has_stop && limit < 0 {
			limit = if limit < -length { -1 } else { limit + length }
		}
		first = clamp_index(first, -1, length - 1)
		limit = clamp_index(limit, -1, length - 1)
		distance := first - limit
		count := if distance == 0 {
			0
		} else if index.step == min_int {
			1
		} else {
			stride := -index.step
			distance / stride + if distance % stride == 0 { 0 } else { 1 }
		}
		return AxisRange{
			start:  first
			length: count
			step:   index.step
		}
	}
}

fn clamp_index(value int, min int, max int) int {
	return if value < min {
		min
	} else if value > max {
		max
	} else {
		value
	}
}
