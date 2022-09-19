module autograd

import vtl
import vtl.la

// add Adds two variables together.
pub fn (v &Variable<T>) add<T>(other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.add<T>(other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := add_gate<T>()
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// subtract Subtracts two variables.
pub fn (v &Variable<T>) subtract<T>(other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.subtract<T>(other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := subtract_gate<T>()
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// multiply Multiplies two variables.
pub fn (v &Variable<T>) multiply<T>(other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.multiply<T>(other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := multiply_gate<T>(v, other)
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// divide Divides two variables.
pub fn (v &Variable<T>) divide<T>(other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.divide<T>(other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := divide_gate<T>(v, other)
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// pow raises a variable to a power.
pub fn (v &Variable<T>) pow<T>(other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.pow<T>(other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := pow_gate<T>(v, other)
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// exp Exponentiates a variable.
pub fn (v &Variable<T>) exp<T>() ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.exp<T>())

	if v.requires_grad {
		gate := exp_gate<T>(v)
		gate.cache<T>(mut result, v)?
	}

	return result
}

// matmul Multiplies two matrices.
pub fn (v &Variable<T>) matmul<T>(other &Variable<T>) ?&Variable<T> {
	mut result := v.context.variable<T>(la.matmul<T>(v.value, other.value)?)

	if v.requires_grad || other.requires_grad {
		gate := matmul_gate<T>(v, other)
		gate.cache<T>(mut result, v, other)?
	}

	return result
}

// sin Sine of a variable.
pub fn (v &Variable<T>) sin<T>() ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.sin<T>())

	if v.requires_grad {
		gate := sin_gate<T>(v)
		gate.cache<T>(mut result, v)?
	}

	return result
}

// cos Cosine of a variable.
pub fn (v &Variable<T>) cos<T>() ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.cos<T>())

	if v.requires_grad {
		gate := cos_gate<T>(v)
		gate.cache<T>(mut result, v)?
	}

	return result
}

// tan Tan of a variable.
pub fn (v &Variable<T>) tan<T>() ?&Variable<T> {
	mut result := v.context.variable<T>(v.value.tan<T>())

	if v.requires_grad {
		gate := tan_gate<T>(v)
		gate.cache<T>(mut result, v)?
	}

	return result
}
