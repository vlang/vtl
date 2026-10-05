import vtl
import vtl.autograd
import vtl.nn.layers
import vtl.nn.models

fn main() {
	ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([1, 5])
	model.conv1d(2, 3, layers.Conv1DConfig{ padding: 1 })
	model.tanh()
	mut sequence := ctx.variable(vtl.from_array([0.1, 0.2, 0.3, 0.4, 0.5], [1, 1, 5]) or {
		eprintln(err)
		return
	})
	mut output := model.forward(sequence) or {
		eprintln(err)
		return
	}
	println('Conv1D output shape: ${output.value.shape}')
	output.backprop() or {
		eprintln(err)
		return
	}
	println('Input gradient shape: ${sequence.grad.shape}')
}
