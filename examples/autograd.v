module main

import vtl
import vtl.autograd

fn main() {
        ctx := autograd.new_ctx<f64>()
        
        x := ctx.variable(vtl.from_1d([3.0]))
        y := ctx.variable(vtl.from_1d([2.0]))

        // mut f := x.pow(y)

        // f.backprop()

        println(x.grad)
}
