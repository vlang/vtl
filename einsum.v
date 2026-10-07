module vtl

import vsl.la as vsl_la
import math.complex as vcomplex

// einsum evaluates an Einstein summation expression over one or more tensors.
// It supports alphabetic axis labels, explicit or implicit output labels,
// repeated labels for diagonals, broadcasting of size-one axes, and any number
// of operands. Ellipses expand to the leading axes and are right-aligned across
// operands, enabling batched expressions such as `...ij,...jk->...ik`.
//
// When the output is omitted, labels that occur exactly once are returned in
// alphabetical order, following NumPy's implicit-output convention.
pub fn einsum[T](subscripts string, operands ...&Tensor[T]) !&Tensor[T] {
	if operands.len == 0 {
		return error('einsum requires at least one operand')
	}
	if subscripts.count('->') > 1 {
		return error('einsum accepts at most one output separator')
	}
	mut input_subscripts := subscripts
	mut output_subscript := ''
	mut has_explicit_output := false
	if arrow := subscripts.index('->') {
		input_subscripts = subscripts[..arrow]
		output_subscript = subscripts[arrow + 2..]
		has_explicit_output = true
	}
	input_terms := input_subscripts.split(',')
	if input_terms.len != operands.len {
		return error('einsum has ${input_terms.len} input terms for ${operands.len} operands')
	}

	mut parsed_operands := []EinsumTerm{cap: operands.len}
	mut ellipsis_rank := 0
	for operand_index, operand in operands {
		parsed := einsum_parse_term(input_terms[operand_index].trim_space()) or { return err }
		if parsed.labels.len > operand.rank() {
			return error('einsum operand ${operand_index} has rank ${operand.rank()} but term has ${parsed.labels.len} labels')
		}
		if !parsed.has_ellipsis && parsed.labels.len != operand.rank() {
			return error('einsum operand ${operand_index} has rank ${operand.rank()} but term has ${parsed.labels.len} labels')
		}
		if parsed.has_ellipsis {
			operand_ellipsis_rank := operand.rank() - parsed.labels.len
			if operand_ellipsis_rank > ellipsis_rank {
				ellipsis_rank = operand_ellipsis_rank
			}
		}
		parsed_operands << parsed
	}
	mut ellipsis_labels := []string{cap: ellipsis_rank}
	for i in 0 .. ellipsis_rank {
		ellipsis_labels << '__vtl_einsum_ellipsis_${i}'
	}
	mut operand_labels := [][]string{cap: operands.len}
	mut label_counts := map[string]int{}
	mut label_sizes := map[string]int{}
	mut labels_in_order := []string{}
	for operand_index, operand in operands {
		labels := einsum_expand_input_labels(parsed_operands[operand_index], operand.rank(),
			ellipsis_labels)
		mut local_sizes := map[string]int{}
		for axis, label in labels {
			dimension := operand.shape[axis]
			if label in local_sizes {
				if local_sizes[label] != dimension {
					return error('einsum repeated label ${label} has incompatible dimensions within operand ${operand_index}')
				}
			} else {
				local_sizes[label] = dimension
			}
			label_counts[label] = label_counts[label] + 1
			if label !in labels_in_order {
				labels_in_order << label
			}
		}
		for label, dimension in local_sizes {
			if label in label_sizes {
				merged_dimension := einsum_broadcast_dimension(label_sizes[label], dimension) or {
					return error('einsum label ${label} has incompatible dimensions ${label_sizes[label]} and ${dimension}')
				}
				label_sizes[label] = merged_dimension
			} else {
				label_sizes[label] = dimension
			}
		}
		operand_labels << labels
	}

	mut output_labels := []string{}
	if has_explicit_output {
		parsed_output := einsum_parse_term(output_subscript.trim_space()) or { return err }
		output_labels = einsum_expand_output_labels(parsed_output, ellipsis_labels)
		mut unique_output_labels := map[string]bool{}
		for label in output_labels {
			if label !in label_sizes {
				return error('einsum output label ${label} does not appear in an input')
			}
			if label in unique_output_labels {
				return error('einsum output label ${label} is repeated')
			}
			unique_output_labels[label] = true
		}
	} else {
		for label in ellipsis_labels {
			output_labels << label
		}
		mut implicit_labels := []string{}
		for label in labels_in_order {
			if label !in ellipsis_labels && label_counts[label] == 1 {
				implicit_labels << label
			}
		}
		implicit_labels.sort()
		output_labels << implicit_labels
	}
	if einsum_is_matrix_multiplication(operand_labels, output_labels)
		&& operands[0].shape[1] == operands[1].shape[0] && operands[0].shape[1] > 0
		&& operands[0].shape[0] > 0 && operands[1].shape[1] > 0 {
		$if T is f32 || T is f64 {
			return einsum_float_matrix_multiplication[T](operands[0], operands[1])
		}
	}

	mut reduction_labels := []string{}
	for label in labels_in_order {
		if label !in output_labels {
			reduction_labels << label
		}
	}
	mut output_shape := []int{cap: output_labels.len}
	for label in output_labels {
		output_shape << label_sizes[label]
	}
	mut reduction_shape := []int{cap: reduction_labels.len}
	for label in reduction_labels {
		reduction_shape << label_sizes[label]
	}

	output_size := einsum_shape_size(output_shape)
	reduction_size := einsum_shape_size(reduction_shape)
	mut result := zeros[T](output_shape)
	mut output_index := []int{len: output_shape.len}
	mut reduction_index := []int{len: reduction_shape.len}
	mut label_positions := map[string]int{}
	for i, label in labels_in_order {
		label_positions[label] = i
	}
	mut label_values := []int{len: labels_in_order.len}
	mut operand_indices := [][]int{cap: operands.len}
	for operand in operands {
		operand_indices << []int{len: operand.rank()}
	}
	for output_linear in 0 .. output_size {
		einsum_fill_index(output_linear, output_shape, mut output_index)
		for i, label in output_labels {
			label_values[label_positions[label]] = output_index[i]
		}
		mut sum := einsum_zero[T]()
		for reduction_linear in 0 .. reduction_size {
			einsum_fill_index(reduction_linear, reduction_shape, mut reduction_index)
			for i, label in reduction_labels {
				label_values[label_positions[label]] = reduction_index[i]
			}
			mut product := einsum_one[T]()
			for operand_index, operand in operands {
				for axis, label in operand_labels[operand_index] {
					operand_indices[operand_index][axis] = if operand.shape[axis] == 1 {
						0
					} else {
						label_values[label_positions[label]]
					}
				}
				product *= operand.get(operand_indices[operand_index])
			}
			sum += product
		}
		result.set(output_index, sum)
	}
	return result
}

fn einsum_zero[T]() T {
	$if T is vcomplex.Complex {
		return T(vcomplex.Complex{})
	} $else {
		return T(0)
	}
}

fn einsum_one[T]() T {
	$if T is vcomplex.Complex {
		return T(vcomplex.Complex{ re: 1, im: 0 })
	} $else {
		return T(1)
	}
}

struct EinsumTerm {
	labels          []string
	has_ellipsis    bool
	ellipsis_offset int
}

fn einsum_is_matrix_multiplication(operand_labels [][]string, output_labels []string) bool {
	if operand_labels.len != 2 || operand_labels[0].len != 2 || operand_labels[1].len != 2
		|| output_labels.len != 2 {
		return false
	}
	return operand_labels[0][1] == operand_labels[1][0]
		&& output_labels[0] == operand_labels[0][0] && output_labels[1] == operand_labels[1][1]
}

fn einsum_float_matrix_multiplication[T](a &Tensor[T], b &Tensor[T]) !&Tensor[T] {
	rows := a.shape[0]
	inner := a.shape[1]
	columns := b.shape[1]
	a_matrix := vsl_la.Matrix.raw(rows, inner, a.to_array().map(f64(it)))
	b_matrix := vsl_la.Matrix.raw(inner, columns, b.to_array().map(f64(it)))
	mut result_matrix := vsl_la.Matrix.new[f64](rows, columns)
	vsl_la.matrix_matrix_mul(mut result_matrix, 1.0, a_matrix, b_matrix)
	mut result := [][]T{cap: rows}
	for row in result_matrix.get_deep2() {
		result << row.map(T(it))
	}
	return from_2d[T](result)
}

fn einsum_parse_term(term string) !EinsumTerm {
	mut labels := []string{cap: term.len}
	mut has_ellipsis := false
	mut ellipsis_offset := 0
	mut i := 0
	for i < term.len {
		character := term[i]
		if character == `.` {
			if has_ellipsis || i + 3 > term.len || term[i..i + 3] != '...' {
				return error('einsum ellipsis must be written as a single `...` token')
			}
			has_ellipsis = true
			ellipsis_offset = labels.len
			i += 3
			continue
		}
		if !((character >= `a` && character <= `z`) || (character >= `A` && character <= `Z`)) {
			return error('einsum labels must be ASCII letters')
		}
		labels << term[i..i + 1]
		i++
	}
	return EinsumTerm{
		labels:          labels
		has_ellipsis:    has_ellipsis
		ellipsis_offset: ellipsis_offset
	}
}

fn einsum_expand_input_labels(term EinsumTerm, rank int, ellipsis_labels []string) []string {
	if !term.has_ellipsis {
		return term.labels.clone()
	}
	ellipsis_count := rank - term.labels.len
	ellipsis_start := ellipsis_labels.len - ellipsis_count
	mut labels := []string{cap: rank}
	labels << term.labels[..term.ellipsis_offset]
	labels << ellipsis_labels[ellipsis_start..]
	labels << term.labels[term.ellipsis_offset..]
	return labels
}

fn einsum_expand_output_labels(term EinsumTerm, ellipsis_labels []string) []string {
	if !term.has_ellipsis {
		return term.labels.clone()
	}
	mut labels := []string{cap: term.labels.len + ellipsis_labels.len}
	labels << term.labels[..term.ellipsis_offset]
	labels << ellipsis_labels
	labels << term.labels[term.ellipsis_offset..]
	return labels
}

fn einsum_broadcast_dimension(a int, b int) !int {
	if a == b {
		return a
	}
	if a == 1 {
		return b
	}
	if b == 1 {
		return a
	}
	return error('incompatible broadcast dimensions')
}

fn einsum_shape_size(shape []int) int {
	mut size := 1
	for dimension in shape {
		size *= dimension
	}
	return size
}

fn einsum_fill_index(linear int, shape []int, mut index []int) {
	mut remaining := linear
	for axis := shape.len - 1; axis >= 0; axis-- {
		index[axis] = remaining % shape[axis]
		remaining /= shape[axis]
	}
}
