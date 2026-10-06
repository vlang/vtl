module main

import vtl

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

	duplicated_time := vtl.repeat_axis(video, 2, 1)!
	println('Repeated time steps [batch, time, channel]: ${duplicated_time.shape}')

	tiled_batch := vtl.tile(video, [2, 1, 1])!
	println('Tiled batch [batch, time, channel]: ${tiled_batch.shape}')

	rotated_batch_time := vtl.rot90(video)!
	println('Rotated batch/time plane: ${rotated_batch_time.shape}')
}
