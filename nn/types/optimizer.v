module types

// Optimizer is a generic interface for all optimizers.
pub interface Optimizer {
	// mut:
	//         params []&autograd.Variable<T>
	//         learning_rate f64
	//         update()
	//         build_params(layer Layer)
}
