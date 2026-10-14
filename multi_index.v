module vtl

// MultiIndexOrder selects row-major or column-major flattening order.
pub enum MultiIndexOrder {
	c
	f
}

// MultiIndexOptions configures ravel_multi_index. Set axis_modes to provide a
// different bounds mode for each coordinate axis; otherwise mode applies to
// every axis.
@[params]
pub struct MultiIndexOptions {
pub:
	mode       TakeMode = .raise
	axis_modes []TakeMode
	order      MultiIndexOrder = .c
}

// UnravelIndexOptions selects the order used by unravel_index. The default is
// C (row-major) order, matching NumPy.
@[params]
pub struct UnravelIndexOptions {
pub:
	order MultiIndexOrder = .c
}

// ravel_multi_index converts one integer coordinate tensor per dimension to
// flat indices. Coordinate tensors broadcast together. The result shape is
// their broadcast shape. `mode` applies raise, wrap, or clip bounds handling
// to every axis; `axis_modes` may instead specify one mode per axis. Order
// selects C (row-major) or F (column-major) flattening.
@[direct_array_access]
pub fn ravel_multi_index(coordinates []&Tensor[int], dimensions []int,
	options MultiIndexOptions) !&Tensor[int] {
	if dimensions.len == 0 || coordinates.len != dimensions.len {
		return error('ravel_multi_index: coordinate count must match a non-empty dimensions list')
	}
	if options.axis_modes.len != 0 && options.axis_modes.len != dimensions.len {
		return error('ravel_multi_index: axis_modes must be empty or contain one mode per dimension')
	}
	total_size := multi_index_total_size(dimensions)!
	broadcast_coordinates := broadcast_n[int](coordinates)!
	output_shape := broadcast_coordinates[0].shape
	output_size := size_from_shape(output_shape)
	if output_size > 0 && total_size == 0 {
		return error('ravel_multi_index: non-empty coordinates cannot index a zero-sized dimension')
	}
	mut output := empty[int](output_shape, memory: .row_major)
	if output_size == 0 {
		return output
	}
	strides := multi_index_strides(dimensions, options.order)
	mut normalized := []int{len: dimensions.len}
	for flat_index in 0 .. output_size {
		for axis, dimension in dimensions {
			mode := if options.axis_modes.len == 0 {
				options.mode
			} else {
				options.axis_modes[axis]
			}
			normalized[axis] = normalize_multi_index_coordinate(broadcast_coordinates[axis].get_nth[int](flat_index),
				dimension, mode) or {
				return error('ravel_multi_index: coordinate on axis ${axis}: ${err.msg()}')
			}
		}
		mut index := 0
		for axis, coordinate in normalized {
			index += coordinate * strides[axis]
		}
		output.data.data[flat_index] = index
	}
	return output
}

// unravel_index converts flat indices into one coordinate tensor per axis.
// Each returned tensor has the same shape as indices. Order selects C
// (row-major) or F (column-major) decoding.
@[direct_array_access]
pub fn unravel_index(indices &Tensor[int], dimensions []int, options UnravelIndexOptions) ![]&Tensor[int] {
	if dimensions.len == 0 {
		return error('unravel_index: dimensions must not be empty')
	}
	total_size := multi_index_total_size(dimensions)!
	if indices.size > 0 && total_size == 0 {
		return error('unravel_index: non-empty indices cannot address a zero-sized shape')
	}
	for flat_position in 0 .. indices.size {
		flat_index := indices.get_nth[int](flat_position)
		if flat_index < 0 || flat_index >= total_size {
			return error('unravel_index: index ${flat_index} is out of bounds for size ${total_size}')
		}
	}
	mut outputs := []&Tensor[int]{cap: dimensions.len}
	for _ in dimensions {
		outputs << empty[int](indices.shape, memory: .row_major)
	}
	if indices.size == 0 {
		return outputs
	}
	strides := multi_index_strides(dimensions, options.order)
	mut coordinate := []int{len: dimensions.len}
	for flat_position in 0 .. indices.size {
		mut remaining := indices.get_nth[int](flat_position)
		if options.order == .c {
			for axis, stride in strides {
				coordinate[axis] = remaining / stride
				remaining %= stride
			}
		} else {
			for axis := dimensions.len - 1; axis >= 0; axis-- {
				coordinate[axis] = remaining / strides[axis]
				remaining %= strides[axis]
			}
		}
		for axis, value in coordinate {
			outputs[axis].data.data[flat_position] = value
		}
	}
	return outputs
}

fn multi_index_total_size(dimensions []int) !int {
	if dimensions.len == 0 {
		return error('multi-index dimensions must not be empty')
	}
	mut has_zero_dimension := false
	for dimension in dimensions {
		if dimension < 0 {
			return error('multi-index dimensions must be non-negative')
		}
		if dimension == 0 {
			has_zero_dimension = true
		}
	}
	if has_zero_dimension {
		return 0
	}
	mut total_size := 1
	for dimension in dimensions {
		if total_size > max_int / dimension {
			return error('multi-index dimensions exceed the maximum tensor size')
		}
		total_size *= dimension
	}
	return total_size
}

fn multi_index_strides(dimensions []int, order MultiIndexOrder) []int {
	mut strides := []int{len: dimensions.len}
	mut stride := 1
	if order == .c {
		for axis := dimensions.len - 1; axis >= 0; axis-- {
			strides[axis] = stride
			stride *= dimensions[axis]
		}
	} else {
		for axis, dimension in dimensions {
			strides[axis] = stride
			stride *= dimension
		}
	}
	return strides
}

fn normalize_multi_index_coordinate(coordinate int, dimension int, mode TakeMode) !int {
	if dimension == 0 {
		return error('cannot index a dimension of size zero')
	}
	match mode {
		.raise {
			if coordinate < 0 || coordinate >= dimension {
				return error('coordinate ${coordinate} is out of bounds for dimension ${dimension}')
			}
			return coordinate
		}
		.wrap {
			wrapped := coordinate % dimension
			return if wrapped < 0 { wrapped + dimension } else { wrapped }
		}
		.clip {
			return if coordinate < 0 {
				0
			} else if coordinate >= dimension {
				dimension - 1
			} else {
				coordinate
			}
		}
	}
}
