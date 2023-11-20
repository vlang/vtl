module types

import vtl.autograd

// Optimizer is a generic interface for all optimizers.
pub interface Optimizer[T] {
mut:
	params        []&autograd.Variable[T]
	learning_rate f64
	update() !
	build_params(layers Layer[T])
}
