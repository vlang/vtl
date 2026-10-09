module la

import vtl

// multi_dot multiplies a sequence of matrices using the parenthesization with
// the lowest estimated scalar multiplication count. Vectors are allowed only
// as the first or last operand, matching NumPy's linalg.multi_dot contract.
pub fn multi_dot[T](operands []&vtl.Tensor[T]) !&vtl.Tensor[T] {
	if operands.len < 2 {
		return error('multi_dot requires at least two operands')
	}
	count := operands.len
	mut dimensions := []int{len: count + 1}
	for i, operand in operands {
		rank := operand.rank()
		is_edge := i == 0 || i == count - 1
		if rank != 2 && !(is_edge && rank == 1) {
			return error('multi_dot accepts matrices and vectors only at the ends')
		}
		rows := if rank == 1 {
			if i == 0 { 1 } else { operand.shape[0] }
		} else {
			operand.shape[rank - 2]
		}
		columns := if rank == 1 {
			if i == count - 1 { 1 } else { operand.shape[0] }
		} else {
			operand.shape[rank - 1]
		}
		if i == 0 {
			dimensions[0] = rows
		} else if dimensions[i] != rows {
			return error('multi_dot dimensions ${dimensions[i - 1]}x${dimensions[i]} and ${rows}x${columns} do not align')
		}
		dimensions[i + 1] = columns
	}
	splits := matrix_chain_splits(dimensions)
	return evaluate_multi_dot[T](operands, splits, 0, count - 1)
}

fn matrix_chain_splits(dimensions []int) []int {
	count := dimensions.len - 1
	mut costs := []f64{len: count * count, init: f64(1e300)}
	mut splits := []int{len: count * count, init: -1}
	for i in 0 .. count {
		costs[i * count + i] = 0
	}
	for chain_length in 2 .. count + 1 {
		for start in 0 .. count - chain_length + 1 {
			end := start + chain_length - 1
			for middle in start .. end {
				cost := costs[start * count + middle] + costs[(middle + 1) * count + end] + f64(dimensions[start]) * f64(dimensions[middle + 1]) * f64(dimensions[end + 1])
				if cost < costs[start * count + end] {
					costs[start * count + end] = cost
					splits[start * count + end] = middle
				}
			}
		}
	}
	return splits
}

fn evaluate_multi_dot[T](operands []&vtl.Tensor[T], splits []int, start int, end int) !&vtl.Tensor[T] {
	if start == end {
		return operands[start]
	}
	count := operands.len
	middle := splits[start * count + end]
	left := evaluate_multi_dot[T](operands, splits, start, middle)!
	right := evaluate_multi_dot[T](operands, splits, middle + 1, end)!
	return matmul[T](left, right)
}
