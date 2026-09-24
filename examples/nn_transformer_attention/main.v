module main

import vtl
import vtl.autograd
import vtl.nn.models

const batch_size = 2
const sequence_length = 4
const embedding_dim = 8
const attention_heads = 2

//runs a small transformer-style self-attention block.
//the input shape is [batch, sequence length, embedding dimension].
fn main() {
	ctx := autograd.ctx[f64]()

	//determinstic  token embeddings make this example easy to reproduce.
	mut values := []f64{cap: batch_size * sequence_length * embedding_dim}
	for batch in 0 .. batch_size {
		for position in 0 .. sequence_length {
			for feature in 0 .. embedding_dim {
				values << f64(batch + 1) * 0.1 + f64(position) * 0.01 + f64(feature) * 0.001
			}
		}
	}
	input_tensor := vtl.from_array(values, [batch_size, sequence_length, embedding_dim])!
	input := ctx.variable(input_tensor)

	mut model := models.sequential_from_ctx[f64](ctx)
	model.input([sequence_length, embedding_dim])
	model.positional_encoding(embedding_dim, sequence_length)
	model.multihead_attention(embedding_dim, attention_heads)

	output := model.forward(input)!

	println('Input shape: ${input.value.shape}')
	println('Output shape: ${output.value.shape}')
	print('First output token: [')
	for feature in 0 .. embedding_dim {
		if feature > 0 {
			print(', ')
		}
		print('${output.value.get_nth(feature):.6f}')
	}
	println(']')
}
