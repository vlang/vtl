module profiler

import time

// TimingEvent records one timed operation.
pub struct TimingEvent {
pub:
	label string
	ns    i64
}

// Profiler collects opt-in timing and CPU memory-accounting events. It does
// not intercept allocations automatically: callers report bytes when they
// allocate and release a reusable buffer, which keeps instrumentation costs
// explicit and attribution precise.
pub struct Profiler {
mut:
	enabled         bool
	active          map[string]i64
	total_active    i64
	peak_active     i64
	total_allocated i64
	timings         []TimingEvent
}

// new returns a profiler. A disabled profiler still runs timed operations but
// records no events, so callers can leave instrumentation in place.
pub fn new(enabled bool) &Profiler {
	return &Profiler{ enabled: enabled }
}

// allocate records bytes held under `label`. Negative sizes are rejected.
pub fn (mut p Profiler) allocate(label string, bytes i64) ! {
	if bytes < 0 {
		return error('allocation bytes must be non-negative')
	}
	if !p.enabled {
		return
	}
	p.active[label] += bytes
	p.total_active += bytes
	p.total_allocated += bytes
	if p.total_active > p.peak_active {
		p.peak_active = p.total_active
	}
}

// release records bytes returned under `label`. Releasing more than the
// currently active bytes for a label is rejected so lifetimes stay balanced.
pub fn (mut p Profiler) release(label string, bytes i64) ! {
	if bytes < 0 {
		return error('release bytes must be non-negative')
	}
	if !p.enabled {
		return
	}
	current := p.active[label]
	if bytes > current {
		return error('released ${bytes} bytes from ${label} but only ${current} are active')
	}
	p.active[label] = current - bytes
	p.total_active -= bytes
}

// active_total_bytes reports bytes currently active across all labels.
pub fn (p &Profiler) active_total_bytes() i64 {
	return p.total_active
}

// timed runs `op`, records its elapsed time under `label`, and returns its
// result. Errors from `op` are propagated and the timing is still recorded.
pub fn timed[T](mut p Profiler, label string, op fn () !T) !T {
	started := time.sys_mono_now()
	result := op()
	elapsed := time.sys_mono_now() - started
	if p.enabled {
		p.timings << TimingEvent{
			label: label
			ns:    elapsed
		}
	}
	return result
}

// peak_active_bytes reports the highest simultaneously active byte count.
pub fn (p &Profiler) peak_active_bytes() i64 {
	return p.peak_active
}

// total_allocated_bytes reports the sum of all reported allocations.
pub fn (p &Profiler) total_allocated_bytes() i64 {
	return p.total_allocated
}

// active_bytes reports bytes currently active for one label.
pub fn (p &Profiler) active_bytes(label string) i64 {
	return p.active[label]
}

// timings returns a copy of the recorded timing events in call order.
pub fn (p &Profiler) timings() []TimingEvent {
	return p.timings.clone()
}

// report returns a short human-readable memory summary.
pub fn (p &Profiler) report() string {
	return 'peak_active_bytes=${p.peak_active} total_allocated_bytes=${p.total_allocated} events=${p.timings.len}'
}

// reset clears all recorded events and active buffers.
pub fn (mut p Profiler) reset() {
	p.active = map[string]i64{}
	p.total_active = 0
	p.peak_active = 0
	p.total_allocated = 0
	p.timings = []TimingEvent{}
}

// max_i64_value is the largest i64 value, used for pre-multiplication
// overflow checks in tensor_bytes.
const max_i64_value = i64(9223372036854775807)

// tensor_bytes computes the byte size of a dense tensor shape. Zero-sized
// dimensions are allowed; negative dimensions and overflow are rejected.
pub fn tensor_bytes[T](shape []int) !i64 {
	mut elements := i64(1)
	for dim in shape {
		if dim < 0 {
			return error('shape dimensions must be non-negative')
		}
		if dim != 0 && elements > max_i64_value / i64(dim) {
			return error('shape overflows element count')
		}
		elements *= i64(dim)
	}
	width := i64(sizeof(T))
	if width != 0 && elements > max_i64_value / width {
		return error('shape overflows byte size')
	}
	return elements * width
}
