module profiler

import vtl.profiler as prof

fn test_allocation_accounting_tracks_peak_and_lifetime() ! {
	mut p := prof.new(true)
	p.allocate('weights', 100)!
	p.allocate('activations', 50)!
	assert p.peak_active_bytes() == 150
	assert p.active_bytes('weights') == 100
	assert p.total_allocated_bytes() == 150

	p.release('activations', 50)!
	assert p.active_bytes('activations') == 0
	assert p.peak_active_bytes() == 150
	assert p.total_allocated_bytes() == 150

	p.allocate('activations', 40)!
	p.allocate('activations', 20)!
	assert p.active_bytes('activations') == 60
	assert p.peak_active_bytes() == 160
	assert p.report().contains('peak_active_bytes=160')
}

fn test_release_more_than_active_is_rejected() ! {
	mut p := prof.new(true)
	p.allocate('weights', 10)!
	p.release('weights', 11) or {
		assert err.msg().contains('only 10 are active')
		return
	}
	assert false, 'over-release should be rejected'
}

fn test_negative_sizes_are_rejected() ! {
	mut p := prof.new(true)
	p.allocate('weights', -1) or {
		assert err.msg().contains('non-negative')
		return
	}
	assert false, 'negative allocation should be rejected'
}

fn test_disabled_profiler_runs_operations_without_recording() ! {
	mut p := prof.new(false)
	value := prof.timed[int](mut p, 'sum', fn () !int {
		return 1 + 2
	})!
	assert value == 3
	assert p.timings().len == 0
	assert p.peak_active_bytes() == 0

	p.allocate('weights', 100)!
	assert p.active_bytes('weights') == 0
	p.release('weights', 100)!
}

fn test_timed_records_events_and_propagates_errors() ! {
	mut p := prof.new(true)
	mut calls := [0]
	first := prof.timed[int](mut p, 'forward', fn [mut calls] () !int {
		calls[0]++
		return 7
	})!
	assert first == 7
	assert calls[0] == 1

	prof.timed[int](mut p, 'backward', fn () !int {
		return error('boom')
	}) or {
		assert err.msg().contains('boom')
		return
	}
	assert false, 'timed should propagate operation errors'
}

fn test_timed_records_erroring_operations() ! {
	mut p := prof.new(true)
	prof.timed[int](mut p, 'failing', fn () !int {
		return error('boom')
	}) or {
		assert err.msg().contains('boom')
	}
	events := p.timings()
	assert events.len == 1
	assert events[0].label == 'failing'
	assert events[0].ns >= 0
}

fn test_reset_clears_state() ! {
	mut p := prof.new(true)
	p.allocate('weights', 10)!
	_ = prof.timed[int](mut p, 'forward', fn () !int {
		return 1
	})!
	p.reset()
	assert p.peak_active_bytes() == 0
	assert p.total_allocated_bytes() == 0
	assert p.active_bytes('weights') == 0
	assert p.timings().len == 0
}

fn test_tensor_bytes_handles_shapes_and_errors() ! {
	assert prof.tensor_bytes[f64]([2, 3])! == 48
	assert prof.tensor_bytes[f32]([4])! == 16
	assert prof.tensor_bytes[f64]([0, 3])! == 0

	prof.tensor_bytes[f64]([-1, 2]) or {
		assert err.msg().contains('non-negative')
		return
	}
	assert false, 'negative dimension should be rejected'
}

fn test_tensor_bytes_rejects_overflow() ! {
	prof.tensor_bytes[f64]([2_000_000_000, 2_000_000_000]) or {
		assert err.msg().contains('overflow')
		return
	}
	assert false, 'overflowing byte size should be rejected'
}

fn test_tensor_bytes_rejects_positive_wrap_overflow() ! {
	// 2e9 * 2e9 * 2 elements (8e18) fit in i64, but the ×8 byte count wraps
	// all the way around to a positive value, so only a pre-multiplication
	// check can catch it.
	prof.tensor_bytes[f64]([2_000_000_000, 2_000_000_000, 2]) or {
		assert err.msg().contains('overflow')
		return
	}
	assert false, 'positive-wrapping byte size should be rejected'
}
