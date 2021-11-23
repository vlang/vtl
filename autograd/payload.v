module autograd

// Payload is a simple wrapper around a Variable.  It
// is only abstracted out to be a bit more explicit that
// it is being passed around through an operation
[heap]
pub struct Payload<T> {
pub:
	// Contents of the paylod
	variable &Variable<T>
}

pub fn new_payload<T>(variable &Variable<T>) &Payload<T> {
	return &Payload<T>{
		variable: variable
	}
}
