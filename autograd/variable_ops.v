module autograd

pub fn (v &Variable[T]) add(other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable(v.value.add(other.value)!)
	if v.requires_grad || other.requires_grad {
		gate := &AddGate[T]{}
		gate.cache(mut result, v, other)!
	}
	return result
}

pub fn (v &Variable[T]) subtract(other &Variable[T]) !&Variable[T] {
	mut result := v.context.variable(v.value.subtract(other.value)!)
	if v.requires_grad || other.requires_grad {
		gate := &SubtractGate[T]{}
		gate.cache(mut result, v, other)!
	}
	return result
}

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
