module datasets

import vtl

struct CounterStreamDataset {
	start   int
	limit   int
	fail_at int = -1
mut:
	next_value int
}

fn (mut source CounterStreamDataset) next_sample() !StreamSample[f64] {
	if source.next_value == source.fail_at {
		return error('stream source failed')
	}
	if source.next_value >= source.limit {
		return StreamSample[f64]{
			finished: true
		}
	}
	value := source.start + source.next_value
	source.next_value++
	return StreamSample[f64]{
		features: vtl.from_1d([f64(value), f64(value + 1)])!
		labels:   vtl.from_1d([f64(value * 10)])!
	}
}

fn (mut source CounterStreamDataset) reset() ! {
	source.next_value = 0
}

fn test_iterable_data_loader_streams_batches_and_replays() ! {
	mut source := &CounterStreamDataset{
		limit: 3
	}
	mut loader := new_iterable_data_loader[f64](source, 2, false)!

	first := loader.next_batch()!
	assert first.available
	assert first.features.shape == [2, 2]
	assert first.features.to_array() == [0.0, 1, 1, 2]
	assert first.labels.to_array() == [0.0, 10]

	last := loader.next_batch()!
	assert last.available
	assert last.features.shape == [1, 2]
	assert last.features.to_array() == [2.0, 3]
	assert last.labels.to_array() == [20.0]
	assert !loader.next_batch()!.available

	loader.reset()!
	replayed := loader.next_batch()!
	assert replayed.available
	assert replayed.features.to_array() == first.features.to_array()
}

fn test_iterable_data_loader_keeps_source_states_independent() ! {
	mut first_source := &CounterStreamDataset{
		start: 0
		limit: 2
	}
	mut second_source := &CounterStreamDataset{
		start: 10
		limit: 2
	}
	mut first_loader := new_iterable_data_loader[f64](first_source, 1, false)!
	mut second_loader := new_iterable_data_loader[f64](second_source, 1, false)!

	first := first_loader.next_batch()!
	second := second_loader.next_batch()!
	assert first.features.to_array() == [0.0, 1]
	assert second.features.to_array() == [10.0, 11]
	assert first_loader.next_batch()!.features.to_array() == [1.0, 2]
	assert second_loader.next_batch()!.features.to_array() == [11.0, 12]
}

fn test_iterable_data_loader_snapshots_reused_sample_buffers() ! {
	mut source := &ReusingStreamDataset{
		sample: vtl.from_1d([0.0])!
	}
	mut loader := new_iterable_data_loader[f64](source, 2, false)!
	batch := loader.next_batch()!
	assert batch.available
	assert batch.features.to_array() == [0.0, 1]
}

fn test_iterable_data_loader_drops_incomplete_batch_and_propagates_errors() ! {
	mut source := &CounterStreamDataset{
		limit: 3
	}
	mut loader := new_iterable_data_loader[f64](source, 2, true)!
	assert loader.next_batch()!.available
	assert !loader.next_batch()!.available

	mut failing_source := &CounterStreamDataset{
		limit:   3
		fail_at: 1
	}
	mut failing_loader := new_iterable_data_loader[f64](failing_source, 2, false)!
	failing_loader.next_batch() or {
		assert err.msg() == 'stream source failed'
		return
	}
	assert false, 'source errors must propagate from next_batch'
}

fn test_iterable_data_loader_rejects_invalid_batch_size_and_shapes() ! {
	mut source := &CounterStreamDataset{
		limit: 1
	}
	new_iterable_data_loader[f64](source, 0, false) or {
		assert err.msg().contains('batch_size')
		return
	}
	assert false, 'non-positive batch sizes must fail'

	bad_shape_source := &MismatchedShapeStreamDataset{}
	mut loader := new_iterable_data_loader[f64](bad_shape_source, 2, false)!
	loader.next_batch() or {
		assert err.msg().contains('inconsistent shape')
		return
	}
	assert false, 'inconsistent sample shapes must fail'
}

struct MismatchedShapeStreamDataset {
mut:
	next_value int
}

fn (mut source MismatchedShapeStreamDataset) next_sample() !StreamSample[f64] {
	if source.next_value == 2 {
		return StreamSample[f64]{
			finished: true
		}
	}
	length := source.next_value + 1
	source.next_value++
	return StreamSample[f64]{
		features: vtl.zeros[f64]([length])
	}
}

fn (mut source MismatchedShapeStreamDataset) reset() ! {
	source.next_value = 0
}

struct ReusingStreamDataset {
	sample &vtl.Tensor[f64]
mut:
	next_value int
}

fn (mut source ReusingStreamDataset) next_sample() !StreamSample[f64] {
	if source.next_value == 2 {
		return StreamSample[f64]{
			finished: true
		}
	}
	source.sample.set_nth(0, f64(source.next_value))
	source.next_value++
	return StreamSample[f64]{
		features: source.sample
	}
}

fn (mut source ReusingStreamDataset) reset() ! {
	source.next_value = 0
}
