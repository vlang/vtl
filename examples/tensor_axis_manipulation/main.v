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
}
