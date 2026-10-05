import vtl
import vtl.autograd
import vtl.nn.models

fn main() {
	ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([3, 1, 2])
	model.gru(2, 4)
	sequence := vtl.from_array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [3, 1, 2]) or {
		eprintln(err)
		return
	}
	mut input := ctx.variable(sequence)
	mut output := model.forward(input) or {
		eprintln(err)
		return
	}
	println('GRU output shape: ${output.value.shape}')
	output.backprop() or {
		eprintln(err)
		return
	}
	println('Input gradient shape: ${input.grad.shape}')
}
