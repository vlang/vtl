module autograd

import vtl

pub struct AddGate<T> {}

pub fn new_add_gate<T>() &AddGate<T> {
        return &AddGate<T>{}
}

pub fn (g &AddGate<T>) backward<T>(payload &Payload<T>) []&vtl.Tensor<T> {
        gradient := payload.variable.grad
        return [gradient, gradient]
}

pub fn (g &AddGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) {
        a := args[0]
        b := args[1]

        if a is Variable<T> && b is Variable<T> {
                result.grad = vtl.zeros_like<T>(result.value)
                result.requires_grad = true

                register<T>("Add", g, result, a, b)
        }
}

pub struct SubstractGate<T> {}

pub fn new_substract_gate<T>() &AddGate<T> {
        return &AddGate<T>{}
}

pub fn (g &SubstractGate<T>) backward<T>(payload &Payload<T>) []&vtl.Tensor<T> {
        gradient := payload.variable.grad
        oposite := vtl.multiply_scalar<T>(gradient, T(-1))
        return [gradient, oposite]
}

pub fn (g &SubstractGate<T>) cache<T>(mut result Variable<T>, args ...CacheParam) {
        a := args[0]
        b := args[1]

        if a is Variable<T> && b is Variable<T> {
                result.grad = vtl.zeros_like<T>(result.value)
                result.requires_grad = true

                register<T>("Sub", g, result, a, b)
        }
}
