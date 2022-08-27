module autograd

import vtl
import vtl.la

// add Adds two variables together.
pub fn add<T>(v &Variable<T>, other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.add<T>(other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := new_add_gate<T>()
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// substract Subtracts two variables.
pub fn substract<T>(v &Variable<T>, other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.substract<T>(other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := new_substract_gate<T>()
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// multiply Multiplies two variables.
pub fn multiply<T>(v &Variable<T>, other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.multiply<T>(other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := new_multiply_gate<T>(v, other)
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// divide Divides two variables.
pub fn divide<T>(v &Variable<T>, other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.divide<T>(other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := new_divide_gate<T>(v, other)
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// pow raises a variable to a power.
pub fn pow<T>(v &Variable<T>, other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.pow<T>(other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := new_pow_gate<T>(v, other)
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// exp Exponentiates a variable.
pub fn exp<T>(v &Variable<T>, other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.exp<T>())

	if v.requires_grad {
		gate := new_exp_gate<T>(v)
		gate.cache<T>(mut result, v)?
	}

	return result
}

// matmul Multiplies two matrices.
pub fn matmul<T>(v &Variable<T>, other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(la.matmul<T>(v.value, other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := new_matmul_gate<T>(v, other)
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// sin Sine of a variable.
pub fn sin<T>(v &Variable<T>, other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.sin<T>())

	if v.requires_grad || other.requires_grad {
		gate := new_sin_gate<T>(v, other)
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// cos Cosine of a variable.
pub fn cos<T>(v &Variable<T>, other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.cos<T>())

	if v.requires_grad || other.requires_grad {
		gate := new_cos_gate<T>(v, other)
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// tan Tan of a variable.
pub fn tan<T>(v &Variable<T>, other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.tan<T>())

	if v.requires_grad || other.requires_grad {
		gate := new_tan_gate<T>(v, other)
		gate.cache<T>(mut result, v, other)?
	}

	return result
}
