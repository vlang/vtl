module autograd_tests

import vtl
import vtl.autograd

fn test_f32_add_backprop() ! {
	mut gctx := autograd.ctx[f32]()
	a := gctx.variable(vtl.ones[f32]([2, 2]))
	b := gctx.variable(vtl.ones[f32]([2, 2]))
	mut c := a.add(b)!
	c.backprop()!
}
