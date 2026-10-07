module vtl

// PadMode selects how values outside a tensor's bounds are generated.
pub enum PadMode {
	constant
	edge
	wrap
	reflect
	symmetric
}

// pad adds a before/after width to every dimension. pad_width contains one
// [before, after] pair per axis. Reflect excludes the edge value; symmetric
// includes it. constant_value is used only with .constant.
pub fn pad[T](t &Tensor[T], pad_width [][]int, mode PadMode, constant_value T) !&Tensor[T] {
	if pad_width.len != t.rank() {
		return error('pad: expected ${t.rank()} [before, after] pairs, got ${pad_width.len}')
	}
	if t.rank() == 0 {
		return t.copy(.row_major)
	}
	mut output_shape := []int{len: t.rank()}
	mut output_size := 1
	for axis, width in pad_width {
		if width.len != 2 {
			return error('pad: axis ${axis} width must contain [before, after]')
		}
		before := width[0]
		after := width[1]
		if before < 0 || after < 0 {
			return error('pad: widths must be non-negative')
		}
		dimension := t.shape[axis]
		if before > max_int - dimension || after > max_int - dimension - before {
			return error('pad: output dimension overflows int on axis ${axis}')
		}
		padded_dimension := dimension + before + after
		output_shape[axis] = padded_dimension
		if padded_dimension != 0 && output_size > max_int / padded_dimension {
			return error('pad: output tensor size overflows int')
		}
		output_size *= padded_dimension
	}
	mut result := empty[T](output_shape, memory: .row_major)
	if result.size == 0 {
		return result
	}
	if t.size == 0 {
		if mode != .constant {
			return error('pad: ${mode} mode requires non-empty input dimensions')
		}
		for index in 0 .. result.size {
			result.set_nth(index, constant_value)
		}
		return result
	}
	mut output_index := []int{len: t.rank()}
	mut input_index := []int{len: t.rank()}
	for flat_index in 0 .. result.size {
		decode_pad_index(flat_index, output_shape, mut output_index)
		mut use_constant := false
		for axis, coordinate in output_index {
			shifted := coordinate - pad_width[axis][0]
			dimension := t.shape[axis]
			if shifted >= 0 && shifted < dimension {
				input_index[axis] = shifted
				continue
			}
			match mode {
				.constant {
					use_constant = true
				}
				.edge {
					input_index[axis] = if shifted < 0 { 0 } else { dimension - 1 }
				}
				.wrap {
					input_index[axis] = positive_pad_mod(shifted, dimension)
				}
				.reflect {
					if dimension == 1 {
						input_index[axis] = 0
					} else {
						period := 2 * (dimension - 1)
						reflected := positive_pad_mod(shifted, period)
						input_index[axis] = if reflected < dimension {
							reflected
						} else {
							period - reflected
						}
					}
				}
				.symmetric {
					period := 2 * dimension
					reflected := positive_pad_mod(shifted, period)
					input_index[axis] = if reflected < dimension {
						reflected
					} else {
						period - 1 - reflected
					}
				}
			}
		}
		result.set_nth(flat_index, if use_constant { constant_value } else { t.get(input_index) })
	}
	return result
}

fn decode_pad_index(flat_index int, shape []int, mut index []int) {
	mut remainder := flat_index
	for axis := shape.len - 1; axis >= 0; axis-- {
		index[axis] = remainder % shape[axis]
		remainder /= shape[axis]
	}
}

fn positive_pad_mod(value int, modulus int) int {
	remainder := value % modulus
	return if remainder < 0 { remainder + modulus } else { remainder }
}
