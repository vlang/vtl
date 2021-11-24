module autograd

// Node is a member of a computational graph that contains
// a reference to a gate, as well as the parents of the operation
// and the payload that resulted from the operation.
[heap]
pub struct Node<T> {
pub:
	// A Gate containing a backwards and cache function for
	// a node
	gate Gate<T>
pub mut:
	// The variables that created this node
	parents []&Variable<T>
	// Wrapper around a Tensor, contains operation data
	payload &Payload<T>
	// Debug use only, contains a name for a node
	name string
}

// new_node
pub fn new_node<T>(gate Gate<T>, parents []&Variable<T>, payload &Payload<T>, name string) &Node<T> {
	return &Node<T>{
		gate: gate
		parents: parents
		payload: payload
		name: name
	}
}
