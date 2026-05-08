module autograd

import vtl.la

// add Adds two variables together.
pub fn (v &Variable[T]) add[T](other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable[T](v.value.add[T](other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := add_gate[T]()
		gate.cache(mut result, v, other)!
	}

	return result
}

// subtract Subtracts two variables.
pub fn (v &Variable[T]) subtract[T](other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable[T](v.value.subtract[T](other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := subtract_gate[T]()
		gate.cache(mut result, v, other)!
	}

	return result
}

// multiply Multiplies two variables.
pub fn (v &Variable[T]) multiply[T](other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable[T](v.value.multiply[T](other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := multiply_gate[T](v, other)
		gate.cache(mut result, v, other)!
	}

	return result
}

// divide Divides two variables.
pub fn (v &Variable[T]) divide[T](other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable[T](v.value.divide[T](other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := divide_gate[T](v, other)
		gate.cache(mut result, v, other)!
	}

	return result
}

// pow raises a variable to a power.
pub fn (v &Variable[T]) pow[T](other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable[T](v.value.pow[T](other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := pow_gate[T](v, other)
		gate.cache(mut result, v, other)!
	}

	return result
}

// exp Exponentiates a variable.
pub fn (v &Variable[T]) exp[T]() !&Variable[T] {
	mut result := v.context.variable[T](v.value.exp[T]())

	if v.requires_grad {
		gate := exp_gate[T](v)
		gate.cache(mut result, v)!
	}

	return result
}

// matmul Multiplies two matrices.
pub fn (v &Variable[T]) matmul[T](other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable[T](la.matmul[T](v.value, other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := matmul_gate[T](v, other)
		gate.cache(mut result, v, other)!
	}

	return result
}

// sin Sine of a variable.
pub fn (v &Variable[T]) sin[T]() !&Variable[T] {
	mut result := v.context.variable[T](v.value.sin[T]())

	if v.requires_grad {
		gate := sin_gate[T](v)
		gate.cache(mut result, v)!
	}

	return result
}

// cos Cosine of a variable.
pub fn (v &Variable[T]) cos[T]() !&Variable[T] {
	mut result := v.context.variable[T](v.value.cos[T]())

	if v.requires_grad {
		gate := cos_gate[T](v)
		gate.cache(mut result, v)!
	}

	return result
}

// tan Tan of a variable.
pub fn (v &Variable[T]) tan[T]() !&Variable[T] {
	mut result := v.context.variable[T](v.value.tan[T]())

	if v.requires_grad {
		gate := tan_gate[T](v)
		gate.cache(mut result, v)!
	}

	return result
}

// log computes the natural logarithm of the variable element-wise.
// Backward: grad * (1 / x)
// Note: inputs must be positive; behaviour for x <= 0 is undefined.
//
// Example:
// ```v
// x := ctx.variable(vtl.from_1d[f64]([1.0, math.e, math.exp(2.0)]))
// y := x.log[f64]()!
// // y.value ≈ [0.0, 1.0, 2.0]
// ```
pub fn (v &Variable[T]) log[T]() !&Variable[T] {
	g := log_gate[T](v)
	t := v.value.log[T]()
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

// abs_op computes the absolute value of the variable element-wise.
// Backward: grad * sign(x)  (0 at x=0)
//
// Named `abs_op` (not `abs`) to avoid collision with the built-in `abs` method
// on `Tensor[T]` which does not participate in the autograd graph.
//
// Example:
// ```v
// x := ctx.variable(vtl.from_1d[f64]([-3.0, 0.0, 4.0]))
// y := x.abs_op[f64]()!
// // y.value = [3.0, 0.0, 4.0]
// ```
pub fn (v &Variable[T]) abs_op[T]() !&Variable[T] {
	g := abs_gate[T](v)
	t := v.value.abs[T]()
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

// sqrt_op computes the element-wise square root of the variable.
// Backward: grad * (1 / (2 * sqrt(x)))
// Note: inputs must be non-negative.
//
// Named `sqrt_op` (not `sqrt`) to avoid collision with `Tensor.sqrt` which
// does not participate in the autograd graph.
//
// Example:
// ```v
// x := ctx.variable(vtl.from_1d[f64]([1.0, 4.0, 9.0]))
// y := x.sqrt_op[f64]()!
// // y.value = [1.0, 2.0, 3.0]
// ```
pub fn (v &Variable[T]) sqrt_op[T]() !&Variable[T] {
	g := sqrt_gate[T](v)
	t := v.value.sqrt[T]()
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

// tanh_op computes the element-wise hyperbolic tangent of the variable.
// Backward: grad * (1 - tanh²(x))
//
// Named `tanh_op` (not `tanh`) to avoid collision with `Tensor.tanh` which
// does not participate in the autograd graph.
//
// Example:
// ```v
// x := ctx.variable(vtl.from_1d[f64]([0.0, 1.0, -1.0]))
// y := x.tanh_op[f64]()!
// // y.value ≈ [0.0, 0.762, -0.762]
// ```
pub fn (v &Variable[T]) tanh_op[T]() !&Variable[T] {
	t := v.value.tanh[T]()
	g := tanh_gate[T](t)
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

// clamp clips the variable element-wise to the range [min_val, max_val].
// Backward: grad is passed through where min_val < x < max_val, zero otherwise.
//
// Example:
// ```v
// x := ctx.variable(vtl.from_1d[f64]([-2.0, 0.5, 3.0]))
// y := x.clamp[f64](-1.0, 1.0)!
// // y.value = [-1.0, 0.5, 1.0]
// ```
pub fn (v &Variable[T]) clamp[T](min_val T, max_val T) !&Variable[T] {
	g := clamp_gate[T](min_val, max_val, v.value)
	t := v.value.map(fn [min_val, max_val] [T](x T, _ []int) T {
		$if T is f64 || T is f32 || T is i16 || T is i8 || T is int {
			return if x < min_val {
				min_val
			} else if x > max_val {
				max_val
			} else {
				x
			}
		} $else {
			return x
		}
	})
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

// reshape returns a new variable with the same data but a different shape.
// The total number of elements must be preserved.
// Backward: gradient is reshaped back to the original shape.
//
// Example:
// ```v
// x := ctx.variable(vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0], [2, 2]))
// y := x.reshape[f64]([4])!
// // y.value.shape = [4]
// ```
pub fn (v &Variable[T]) reshape[T](new_shape []int) !&Variable[T] {
	g := reshape_gate[T](v.value.shape)
	t := v.value.reshape[T](new_shape)!
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

// transpose_op permutes the axes of the variable according to `perm`.
// `perm` must be a permutation of [0, 1, ..., ndim-1].
// Backward: gradient is transposed with the inverse permutation.
//
// Named `transpose_op` (not `transpose`) to avoid collision with `Tensor.transpose`.
//
// Example:
// ```v
// x := ctx.variable(vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]))
// y := x.transpose_op[f64]([1, 0])!
// // y.value.shape = [3, 2]
// ```
pub fn (v &Variable[T]) transpose_op[T](perm []int) !&Variable[T] {
	g := transpose_gate[T](perm)
	t := v.value.transpose(perm)!
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}
