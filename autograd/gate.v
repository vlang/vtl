module autograd

import vtl

pub interface CacheParam {}

// Gate is an object that can cache the result of an operation,
// as well as backpropogate a payload backwards along the
// computational graph
//
// Structs that implement from this interface can add instance
// variables if additional caching is needed, and these need
// to be populated when writing the cached operation
// @todo: Make this generic once it works as expected
pub interface Gate {
	// backward(payload &Payload<T>) []&vtl.Tensor<T>
	// cache(mut result Variable<T>, args ...CacheParam)
}

// @todo: Implement this somehow :D
pub fn gate_backward[T](gate Gate, payload &Payload[T]) ![]&vtl.Tensor[T] {
	match gate {
		AddGate[T] {
			return gate.backward[T](payload)
		}
		SubstractGate[T] {
			return gate.backward[T](payload)
		}
		MultiplyGate[T] {
			return gate.backward[T](payload)
		}
		DivideGate[T] {
			return gate.backward[T](payload)
		}
		MatMulGate[T] {
			return gate.backward[T](payload)
		}
		ExpGate[T] {
			return gate.backward[T](payload)
		}
		PowGate[T] {
			return gate.backward[T](payload)
		}
		SinGate[T] {
			return gate.backward[T](payload)
		}
		CosGate[T] {
			return gate.backward[T](payload)
		}
		TanGate[T] {
			return gate.backward[T](payload)
		}
		else {
			return error(@FN + ' is not supported for type ${typeof(gate).name}')
		}
	}
}

// @todo: Implement this somehow :D
pub fn gate_cache[T](gate Gate, mut result Variable[T], args ...CacheParam) ! {
	match gate {
		AddGate[T] {
			return gate.cache[T](mut result, ...args)
		}
		SubstractGate[T] {
			return gate.cache[T](mut result, ...args)
		}
		MultiplyGate[T] {
			return gate.cache[T](mut result, ...args)
		}
		DivideGate[T] {
			return gate.cache[T](mut result, ...args)
		}
		MatMulGate[T] {
			return gate.cache[T](mut result, ...args)
		}
		ExpGate[T] {
			return gate.cache[T](mut result, ...args)
		}
		PowGate[T] {
			return gate.cache[T](mut result, ...args)
		}
		SinGate[T] {
			return gate.cache[T](mut result, ...args)
		}
		CosGate[T] {
			return gate.cache[T](mut result, ...args)
		}
		TanGate[T] {
			return gate.cache[T](mut result, ...args)
		}
		else {
			return error(@FN + ' is not supported for type ${typeof(gate).name}')
		}
	}
}
