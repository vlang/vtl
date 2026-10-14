module autograd

import vtl
import math.complex as vcomplex
import vtl.stats as vtl_stats

// sum reduces every tensor element into a one-element tensor and records the
// broadcast backward rule when this variable requires gradients.
pub fn (v &Variable[T]) sum() !&Variable[T] {
	value := vtl.from_1d([vtl_stats.sum[T](v.value)])!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := sum_gate[T](v.value.shape, -1)
		gate.cache(mut result, v)!
	}
	return result
}

// mean reduces every tensor element into a one-element tensor and distributes
// the backward gradient uniformly over the input elements.
pub fn (v &Variable[T]) mean() !&Variable[T] {
	value := vtl.from_1d([vtl_stats.mean[T](v.value)])!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := mean_gate[T](v.value.shape, -1, v.value.size)
		gate.cache(mut result, v)!
	}
	return result
}

// sum_along_axis reduces one axis. Set keepdims to retain it with length one.
pub fn (v &Variable[T]) sum_along_axis(axis int, keepdims bool) !&Variable[T] {
	rank := v.value.rank()
	if rank == 0 {
		return error('Variable.sum_along_axis: input has no dimensions')
	}
	na := if axis < 0 { axis + rank } else { axis }
	if na < 0 || na >= rank {
		return error('Variable.sum_along_axis: axis ${axis} out of bounds for shape ${v.value.shape}')
	}
	value := vtl_stats.sum_along_axis[T](v.value, na, keepdims)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := sum_gate[T](v.value.shape, na)
		gate.cache(mut result, v)!
	}
	return result
}

// cumsum applies a cumulative sum along one axis and records its backward
// rule when this variable requires gradients.
pub fn (v &Variable[T]) cumsum(axis int) !&Variable[T] {
	rank := v.value.rank()
	if rank == 0 {
		return error('Variable.cumsum: input has no dimensions')
	}
	na := if axis < 0 { axis + rank } else { axis }
	if na < 0 || na >= rank {
		return error('Variable.cumsum: axis ${axis} out of bounds for shape ${v.value.shape}')
	}
	value := v.value.cumsum[T](na)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := cumsum_gate[T](v.value.shape, na)
		gate.cache(mut result, v)!
	}
	return result
}

// cumprod applies a cumulative product along one axis and records its backward
// rule when this variable requires gradients.
pub fn (v &Variable[T]) cumprod(axis int) !&Variable[T] {
	rank := v.value.rank()
	if rank == 0 {
		return error('Variable.cumprod: input has no dimensions')
	}
	na := if axis < 0 { axis + rank } else { axis }
	if na < 0 || na >= rank {
		return error('Variable.cumprod: axis ${axis} out of bounds for shape ${v.value.shape}')
	}
	value := v.value.cumprod[T](na)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := cumprod_gate[T](v.value, na)
		gate.cache(mut result, v)!
	}
	return result
}

// mean_along_axis computes one-axis means. Set keepdims to retain the reduced
// axis with length one.
pub fn (v &Variable[T]) mean_along_axis(axis int, keepdims bool) !&Variable[T] {
	rank := v.value.rank()
	if rank == 0 {
		return error('Variable.mean_along_axis: input has no dimensions')
	}
	na := if axis < 0 { axis + rank } else { axis }
	if na < 0 || na >= rank {
		return error('Variable.mean_along_axis: axis ${axis} out of bounds for shape ${v.value.shape}')
	}
	mean_values := vtl_stats.mean_along_axis[T](v.value, na, keepdims)!
	value := mean_values.map[T](fn [T](mean f64, _ []int) T {
		return vtl.cast[T](mean)
	})
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := mean_gate[T](v.value.shape, na, v.value.shape[na])
		gate.cache(mut result, v)!
	}
	return result
}

// sum_along_axes reduces several axes. Negative axes are accepted; an empty
// axes list returns an elementwise copy. Set keepdims to retain reduced axes.
pub fn (v &Variable[T]) sum_along_axes(axes []int, keepdims bool) !&Variable[T] {
	normalized_axes := normalize_variable_reduction_axes(axes, v.value.rank(), 'sum_along_axes')!
	value := vtl_stats.sum_along_axes[T](v.value, normalized_axes, keepdims)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := multi_axis_reduction_gate[T](v.value.shape, normalized_axes, keepdims, 1, false)
		gate.cache(mut result, v)!
	}
	return result
}

// mean_along_axes computes means over several axes and propagates gradients.
// Negative axes are accepted; an empty axes list returns an elementwise copy.
pub fn (v &Variable[T]) mean_along_axes(axes []int, keepdims bool) !&Variable[T] {
	normalized_axes := normalize_variable_reduction_axes(axes, v.value.rank(), 'mean_along_axes')!
	mean_values := vtl_stats.mean_along_axes[T](v.value, normalized_axes, keepdims)!
	value := mean_values.map[T](fn [T](mean f64, _ []int) T {
		return vtl.cast[T](mean)
	})
	mut num_elems := 1
	for axis in normalized_axes {
		num_elems *= v.value.shape[axis]
	}
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := multi_axis_reduction_gate[T](v.value.shape, normalized_axes, keepdims, num_elems,
			true)
		gate.cache(mut result, v)!
	}
	return result
}

fn normalize_variable_reduction_axes(axes []int, rank int, operation string) ![]int {
	if rank == 0 && axes.len > 0 {
		return error('Variable.${operation}: input has no dimensions')
	}
	mut seen := []bool{len: rank}
	mut normalized := []int{cap: axes.len}
	for axis in axes {
		axis_index := if axis < 0 { axis + rank } else { axis }
		if axis_index < 0 || axis_index >= rank {
			return error('Variable.${operation}: axis ${axis} out of bounds for shape')
		}
		if seen[axis_index] {
			return error('Variable.${operation}: axis ${axis} appears more than once')
		}
		seen[axis_index] = true
		normalized << axis_index
	}
	return normalized
}

// add Adds two variables together.
pub fn (v &Variable[T]) add(other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable(v.value.add(other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := &AddGate[T]{}
		gate.cache(mut result, v, other)!
	}

	return result
}

// subtract Subtracts two variables.
pub fn (v &Variable[T]) subtract(other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable(v.value.subtract(other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := &SubtractGate[T]{}
		gate.cache(mut result, v, other)!
	}

	return result
}

// multiply Multiplies two variables.
pub fn (v &Variable[T]) multiply(other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable(v.value.multiply(other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := &MultiplyGate[T]{
			a: v
			b: other
		}
		gate.cache(mut result, v, other)!
	}

	return result
}

// divide Divides two variables.
pub fn (v &Variable[T]) divide(other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable(v.value.divide(other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := &DivideGate[T]{
			a: v
			b: other
		}
		gate.cache(mut result, v, other)!
	}

	return result
}

// pow raises a variable to a power.
pub fn (v &Variable[T]) pow(other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable(v.value.pow(other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := pow_gate[T](v, other)
		gate.cache(mut result, v, other)!
	}

	return result
}

// exp Exponentiates a variable.
pub fn (v &Variable[T]) exp() !&Variable[T] {
	mut result := v.context.variable(v.value.exp())

	if v.requires_grad {
		gate := exp_gate[T](v)
		gate.cache(mut result, v)!
	}

	return result
}

// matmul Multiplies two matrices.
pub fn (v &Variable[T]) matmul(other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable(gate_matmul[T](v.value, other.value)!)

	if v.requires_grad || other.requires_grad {
		gate := &MatMulGate[T]{
			a: v
			b: other
		}
		gate.cache(mut result, v, other)!
	}

	return result
}

// sin Sine of a variable.
pub fn (v &Variable[T]) sin() !&Variable[T] {
	mut result := v.context.variable(v.value.sin())

	if v.requires_grad {
		gate := sin_gate[T](v)
		gate.cache(mut result, v)!
	}

	return result
}

// cos Cosine of a variable.
pub fn (v &Variable[T]) cos() !&Variable[T] {
	mut result := v.context.variable(v.value.cos())

	if v.requires_grad {
		gate := cos_gate[T](v)
		gate.cache(mut result, v)!
	}

	return result
}

// tan Tan of a variable.
pub fn (v &Variable[T]) tan() !&Variable[T] {
	mut result := v.context.variable(v.value.tan())

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
pub fn (v &Variable[T]) log() !&Variable[T] {
	g := log_gate[T](v)
	t := v.value.log()
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
pub fn (v &Variable[T]) abs_op() !&Variable[T] {
	g := abs_gate[T](v)
	t := v.value.abs()
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
pub fn (v &Variable[T]) sqrt_op() !&Variable[T] {
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
pub fn (v &Variable[T]) tanh_op() !&Variable[T] {
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
pub fn (v &Variable[T]) clamp(min_val T, max_val T) !&Variable[T] {
	g := clamp_gate[T](min_val, max_val, v.value)
	t := v.value.map(fn [min_val, max_val] [T](x T, _ []int) T {
		$if T is f64 || T is f32 || T is i16 || T is i32 || T is i8 || T is int {
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

// nan_to_num replaces NaN and infinite values and passes gradients through
// finite inputs only. Its forward and backward behavior follows PyTorch.
pub fn (v &Variable[T]) nan_to_num(params vtl.NanToNumConfig) !&Variable[T] {
	$if T is vcomplex.Complex {
		return error('Variable.nan_to_num does not yet support complex tensors')
	}
	value := v.value.nan_to_num[T](params)
	mut result := v.context.variable(value)
	if v.requires_grad {
		mut finite_mask := vtl.ones[bool](v.value.shape)
		$if T is f32 || T is f64 {
			finite_mask = v.value.is_finite[T]()
		}
		gate := nan_to_num_gate[T](finite_mask)
		gate.cache(mut result, v)!
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
pub fn (v &Variable[T]) reshape(new_shape []int) !&Variable[T] {
	g := reshape_gate[T](v.value.shape)
	t := v.value.reshape(new_shape)!
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
pub fn (v &Variable[T]) transpose_op(perm []int) !&Variable[T] {
	g := transpose_gate[T](perm)
	t := v.value.transpose(perm)!
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

// rot90 rotates the first two dimensions counterclockwise and propagates
// gradients through the inverse rotation.
pub fn (v &Variable[T]) rot90() !&Variable[T] {
	return v.rot90_axes(1, [0, 1])
}

// rot90_k rotates the first two dimensions counterclockwise by k quarter turns.
pub fn (v &Variable[T]) rot90_k(k int) !&Variable[T] {
	return v.rot90_axes(k, [0, 1])
}

// rot90_axes rotates in a selected plane and records its inverse for autograd.
pub fn (v &Variable[T]) rot90_axes(k int, axes []int) !&Variable[T] {
	value := vtl.rot90_axes[T](v.value, k, axes)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := rot90_gate[T](v.value, k, axes)!
		gate.cache(mut result, v)!
	}
	return result
}

// flip reverses selected axes (all axes by default) and routes gradients
// through the same flip, which is its own inverse.
pub fn (v &Variable[T]) flip(axes ...int) !&Variable[T] {
	value := v.value.flip[T](...axes)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := flip_gate[T](axes)
		gate.cache(mut result, v)!
	}
	return result
}

// repeat flattens the input and repeats every value; backward sums gradients
// from all copies of each source value.
pub fn (v &Variable[T]) repeat(repeats int) !&Variable[T] {
	value := vtl.repeat[T](v.value, repeats)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := repeat_gate[T](v.value.shape, repeats, 0, true)
		gate.cache(mut result, v)!
	}
	return result
}

// repeat_axis repeats values along one axis while preserving autograd.
pub fn (v &Variable[T]) repeat_axis(repeats int, axis int) !&Variable[T] {
	value := vtl.repeat_axis[T](v.value, repeats, axis)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		normalized_axis := if axis < 0 { axis + v.value.rank() } else { axis }
		gate := repeat_gate[T](v.value.shape, repeats, normalized_axis, false)
		gate.cache(mut result, v)!
	}
	return result
}

// tile repeats whole blocks according to NumPy's right-aligned repetition
// dimensions and sums gradients from each block during backpropagation.
pub fn (v &Variable[T]) tile(reps []int) !&Variable[T] {
	value := vtl.tile[T](v.value, reps)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := tile_gate[T](v.value.shape, reps)
		gate.cache(mut result, v)!
	}
	return result
}

// pad adds values around each axis and accumulates gradients from padded
// positions according to the selected padding mode.
pub fn (v &Variable[T]) pad(pad_width [][]int, mode vtl.PadMode, constant_value T) !&Variable[T] {
	value := vtl.pad[T](v.value, pad_width, mode, constant_value)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := pad_gate[T](v.value.shape, pad_width, mode)
		gate.cache(mut result, v)!
	}
	return result
}

// roll circularly shifts values, flattening by default like NumPy. Optional
// axes apply the same shift to each selected axis and preserve autograd.
pub fn (v &Variable[T]) roll(shift int, axes ...int) !&Variable[T] {
	value := vtl.roll[T](v.value, shift, ...axes)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := roll_gate[T](v.value, shift, axes)!
		gate.cache(mut result, v)!
	}
	return result
}

// roll_shifts applies paired shifts and axes while preserving autograd.
pub fn (v &Variable[T]) roll_shifts(shifts []int, axes []int) !&Variable[T] {
	value := vtl.roll_shifts[T](v.value, shifts, axes)!
	mut result := v.context.variable(value)
	if v.requires_grad {
		gate := roll_shifts_gate[T](v.value, shifts, axes)!
		gate.cache(mut result, v)!
	}
	return result
}

// concatenate joins variables along an existing axis and routes gradients back
// to each input. All variables must have compatible shapes and share a context.
pub fn concatenate[T](variables []&Variable[T], data vtl.AxisData) !&Variable[T] {
	if variables.len == 0 {
		return error('cannot concatenate an empty list of variables')
	}
	first := variables[0]
	rank := first.value.rank()
	axis := if data.axis < 0 { data.axis + rank } else { data.axis }
	if axis < 0 || axis >= rank {
		return error('axis out of range')
	}
	mut tensors := []&vtl.Tensor[T]{cap: variables.len}
	mut splits := []int{cap: variables.len}
	mut track_gradient := false
	for variable in variables {
		if variable.context != first.context {
			return error('all variables must share the same autograd context')
		}
		tensors << variable.value
		track_gradient = track_gradient || variable.requires_grad
	}
	value := vtl.concatenate[T](tensors, axis: axis)!
	for variable in variables {
		splits << variable.value.shape[axis]
	}
	mut result := first.context.variable(value, requires_grad: track_gradient)
	if track_gradient {
		gate := concat_gate[T](axis, splits)
		gate.cache(mut result, ...variables)!
	}
	return result
}

// stack inserts a new axis and joins variables along it. Its backward pass
// removes that axis through each input's reshape gate.
pub fn stack[T](variables []&Variable[T], data vtl.AxisData) !&Variable[T] {
	if variables.len == 0 {
		return error('cannot stack an empty list of variables')
	}
	first := variables[0]
	axis := if data.axis < 0 { data.axis + first.value.rank() + 1 } else { data.axis }
	if axis < 0 || axis > first.value.rank() {
		return error('axis out of range')
	}
	mut expanded := []&Variable[T]{cap: variables.len}
	for variable in variables {
		if variable.context != first.context {
			return error('all variables must share the same autograd context')
		}
		if variable.value.shape != first.value.shape {
			return error('all variables must have the same shape to stack')
		}
		mut shape := variable.value.shape.clone()
		shape.insert(axis, 1)
		if variable.requires_grad {
			expanded << variable.reshape(shape)!
		} else {
			expanded << variable.context.variable(variable.value.reshape(shape)!,
				requires_grad: false
			)
		}
	}
	return concatenate[T](expanded, axis: axis)
}
