module main

import vtl
import vtl.autograd

fn main() {
	video := vtl.from_array([]f32{len: 24, init: f32(index)}, [2, 3, 4])!
	println('Video [batch, time, channel]: ${video.shape}')

	batch_last := video.moveaxis([0], [-1])!
	println('Batch last [time, channel, batch]: ${batch_last.shape}')
	println('Logical value at [2, 3, 1]: ${batch_last.get([2, 3, 1])}')

	frames_first := video.rollaxis(2, 0)!
	println('Frames first [channel, batch, time]: ${frames_first.shape}')
	println('Logical value at [3, 1, 2]: ${frames_first.get([3, 1, 2])}')

	last_frames_reversed := video.flip(1)!
	println('Time reversed [batch, time, channel]: ${last_frames_reversed.shape}')
	println('First reversed frame value: ${last_frames_reversed.get([0, 0, 0])}')

	shifted_time := vtl.roll(video, 1, 1)!
	println('Circularly shifted time steps: ${shifted_time.get([0, 0, 0])}, ${shifted_time.get([
		0,
		1,
		0,
	])}, ${shifted_time.get([0, 2, 0])}')

	duplicated_time := vtl.repeat_axis(video, 2, 1)!
	println('Repeated time steps [batch, time, channel]: ${duplicated_time.shape}')

	tiled_batch := vtl.tile(video, [2, 1, 1])!
	println('Tiled batch [batch, time, channel]: ${tiled_batch.shape}')

	rotated_batch_time := vtl.rot90(video)!
	println('Rotated batch/time plane: ${rotated_batch_time.shape}')

	mut context := autograd.ctx[f64]()
	input := context.variable(vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])!)
	weights := context.variable(vtl.from_array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3])!)
	rolled := input.roll(1, 1)!
	mut loss := rolled.multiply(weights)!.sum()!
	loss.backprop()!
	println('Gradient after rolling along columns: ${input.grad.to_array()}')
}
