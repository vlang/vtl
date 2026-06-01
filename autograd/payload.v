module autograd

// Payload is a simple wrapper around a Variable.  It
// is only abstracted out to be a bit more explicit that
// it is being passed around through an operation

// Payload defines a public data structure for this module.

// Payload defines a public data structure for this module.
@[heap]
pub struct Payload[T] {
pub:
	// Contents of the paylod
	variable &Variable[T] = unsafe { nil }
}

// payload exposes this operation as part of the public API.
pub fn payload[T](variable &Variable[T]) &Payload[T] {
	return &Payload[T]{
		variable: variable
	}
}
