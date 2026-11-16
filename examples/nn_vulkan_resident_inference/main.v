import vtl
import vtl.autograd
import vtl.nn.models

fn main() {
	mut ctx := autograd.ctx[f32]()
	mut model := models.sequential_from_ctx[f32](ctx)
	model.input([4])
	model.relu()
	model.sigmoid()
	model.eval()
	ctx.no_grad = true

	input := ctx.variable(vtl.from_array([f32(-2), -0.5, 0.25, 3], [4])!,
		requires_grad: false
	)
	output := model.forward_vulkan_inference(input) or {
		eprintln('Vulkan inference failed: ${err}')
		exit(1)
		return
	}
	println(output.value.to_array())
}
