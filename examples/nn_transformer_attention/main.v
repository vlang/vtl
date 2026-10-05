import rand
import vtl
import vtl.autograd
import vtl.nn.models

fn main() {
	rand.seed([u32(42), u32(0)])
	ctx := autograd.ctx[f64]()
	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([4, 8])
	model.positional_encoding(8, 16)
	model.multihead_attention(8, 2)
	sequence := vtl.from_array([
		0.1,
		0.2,
		0.3,
		0.4,
		0.5,
		0.6,
		0.7,
		0.8,
		0.2,
		0.3,
		0.4,
		0.5,
		0.6,
		0.7,
		0.8,
		0.9,
		0.3,
		0.4,
		0.5,
		0.6,
		0.7,
		0.8,
		0.9,
		1.0,
		0.4,
		0.5,
		0.6,
		0.7,
		0.8,
		0.9,
		1.0,
		1.1,
	], [1, 4, 8]) or {
		eprintln(err)
		return
	}
	input := ctx.variable(sequence)
	mut output := model.forward(input) or {
		eprintln(err)
		return
	}
	println('Input shape: ${input.value.shape}')
	println('Self-attention output shape: ${output.value.shape}')
	println('First output value: ${output.value.get([0, 0, 0])}')
}
