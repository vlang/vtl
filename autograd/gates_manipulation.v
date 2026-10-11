module autograd

import vtl

// PadGate stores the source shape and padding contract needed to route each
// padded output gradient back to its source position.
pub struct PadGate[T] {
pub:
	shape     []int
	pad_width [][]int
	mode      vtl.PadMode
}

// pad_gate creates backward metadata for a padded tensor.
pub fn pad_gate[T](shape []int, pad_width [][]int, mode vtl.PadMode) &PadGate[T] {
	return &PadGate[T]{
		shape:     shape.clone()
		pad_width: pad_width.map(it.clone())
		mode:      mode
	}
}

// backward accumulates gradients according to the forward padding mode.
pub fn (g &PadGate[T]) backward(payload &Payload[T]) ![]&vtl.Tensor[T] {
	gradient := payload.variable.grad
	mut result := vtl.zeros[T](g.shape, vtl.TensorData{})
	if result.size == 0 {
		return [result]
	}
	if g.shape.len == 0 {
		return [gradient.reshape[T](g.shape)!]
	}
	mut output_shape := []int{len: g.shape.len}
	for axis, dim in g.shape {
		output_shape[axis] = dim + g.pad_width[axis][0] + g.pad_width[axis][1]
	}
	mut output_index := []int{len: g.shape.len}
	mut source_index := []int{len: g.shape.len}
	for flat_index in 0 .. gradient.size {
		mut remainder := flat_index
		for axis := output_shape.len - 1; axis >= 0; axis-- {
			output_index[axis] = remainder % output_shape[axis]
			remainder /= output_shape[axis]
		}
		mut use_constant := false
		for axis, coordinate in output_index {
			shifted := coordinate - g.pad_width[axis][0]
			dimension := g.shape[axis]
			if shifted >= 0 && shifted < dimension {
				source_index[axis] = shifted
				continue
			}
			match g.mode {
				.constant {
					use_constant = true
				}
				.edge {
					source_index[axis] = if shifted < 0 { 0 } else { dimension - 1 }
				}
				.wrap {
					source_index[axis] = positive_pad_gradient_mod(shifted, dimension)
				}
				.reflect {
					if dimension == 1 {
						source_index[axis] = 0
					} else {
						period := 2 * (dimension - 1)
						reflected := positive_pad_gradient_mod(shifted, period)
						source_index[axis] = if reflected < dimension {
							reflected
						} else {
							period - reflected
						}
					}
				}
				.symmetric {
					period := 2 * dimension
					reflected := positive_pad_gradient_mod(shifted, period)
					source_index[axis] = if reflected < dimension {
						reflected
					} else {
						period - 1 - reflected
					}
				}
			}
		}
		if use_constant {
			continue
		}
		mut source_flat := 0
		for axis, coordinate in source_index {
			source_flat = source_flat * g.shape[axis] + coordinate
		}
		result.set_nth(source_flat, result.get_nth(source_flat) + gradient.get_nth(flat_index))
	}
	return [result]
}

fn positive_pad_gradient_mod(value int, modulus int) int {
	remainder := value % modulus
	return if remainder < 0 { remainder + modulus } else { remainder }
}

fn pad_gate_backward_dispatch[T](gate voidptr, payload voidptr) ![]voidptr {
	typed_payload := unsafe { &Payload[T](payload) }
	tensors := unsafe { (&PadGate[T](gate)).backward(typed_payload)! }
	return tensor_ptrs_to_voidptrs[T](tensors)
}

// cache registers the input variable for padding backpropagation.
pub fn (g &PadGate[T]) cache(mut result Variable[T], args ...CacheParam) ! {
	a := args[0]
	match a {
		Variable[T] {
			register[T]('Pad', voidptr(g), pad_gate_backward_dispatch[T], result, [a])!
		}
		else {
			return error('PadGate: a must be a Variable')
		}
	}
}
