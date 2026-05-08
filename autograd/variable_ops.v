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

pub fn (v &Variable[T]) log[T]() !&Variable[T] {
	g := log_gate[T](v)
	t := v.value.log[T]()
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

pub fn (v &Variable[T]) abs_op[T]() !&Variable[T] {
	g := abs_gate[T](v)
	t := v.value.abs[T]()
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

pub fn (v &Variable[T]) sqrt_op[T]() !&Variable[T] {
	g := sqrt_gate[T](v)
	t := v.value.sqrt[T]()
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

pub fn (v &Variable[T]) tanh_op[T]() !&Variable[T] {
	t := v.value.tanh[T]()
	g := tanh_gate[T](t)
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

pub fn (v &Variable[T]) clamp[T](min_val T, max_val T) !&Variable[T] {
	g := clamp_gate[T](min_val, max_val, v.value)
	t := v.value.map(fn [min_val, max_val] [T](x T, _ []int) T {
		$if T is f64 || T is f32 || T is i16 || T is i8 || T is int {
			return if x < min_val { min_val } else if x > max_val { max_val } else { x }
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

pub fn (v &Variable[T]) reshape[T](new_shape []int) !&Variable[T] {
	g := reshape_gate[T](v.value.shape)
	t := v.value.reshape[T](new_shape)!
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}

pub fn (v &Variable[T]) transpose_op[T](perm []int) !&Variable[T] {
	g := transpose_gate[T](perm)
	t := v.value.transpose(perm)!
	mut result := v.context.variable(t)
	if v.requires_grad {
		g.cache(mut result, v)!
	}
	return result
}
