module autograd

// Node is a member of a computational graph that contains
// a reference to a gate, as well as the parents of the operation
// and the payload that resulted from the operation.

// Node defines a public data structure for this module.

// Node defines a public data structure for this module.
@[heap]
pub struct Node[T] {
pub:
	// Opaque pointer to the concrete gate instance for this node.
	gate voidptr
	// Callback that knows how to cast and run the gate.
	backward BackwardFn = unsafe { nil }
pub mut:
	// The variables that created this node
	parents []&Variable[T]
	// Wrapper around a Tensor, contains operation data
	payload &Payload[T] = unsafe { nil }
	// Debug use only, contains a name for a node
	name string
}

// node
pub fn node[T](gate voidptr, backward BackwardFn, parents []&Variable[T], payload &Payload[T], name string) &Node[T] {
	return &Node[T]{
		gate:     gate
		backward: backward
		parents:  parents
		payload:  payload
		name:     name
	}
}
