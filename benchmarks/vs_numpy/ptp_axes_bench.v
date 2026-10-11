// Compile with -prod from ~/.vmodules, then run the binary.
module main

import os
import time
import vtl
import vtl.stats

const ptp_axes_width = 256
const ptp_axes_height = 512
const ptp_axes_iterations = 5

fn main() {
	requested_case := os.getenv('VTL_PTP_AXES_BENCH_CASE')
	case_name := if requested_case == '' { 'trailing' } else { requested_case }
	mut axes := [1, 2]
	if case_name == 'interleaved' {
		axes = [0, 2]
	} else if case_name != 'trailing' {
		eprintln('VTL_PTP_AXES_BENCH_CASE must be trailing or interleaved')
		exit(2)
	}
	requested_groups := os.getenv('VTL_PTP_AXES_BENCH_GROUPS')
	groups := if requested_groups == '' { 128 } else { requested_groups.int() }
	if groups <= 0 || groups > max_int / (ptp_axes_width * ptp_axes_height) {
		eprintln('VTL_PTP_AXES_BENCH_GROUPS must be a positive integer within the supported size')
		exit(2)
	}
	size := groups * ptp_axes_width * ptp_axes_height
	mut values := []f64{len: size}
	for i in 0 .. size {
		group_scale := f64(i / (ptp_axes_width * ptp_axes_height) % 8 + 1)
		values[i] = (f64(i % 2001) - 1000.0) * group_scale
	}
	tensor := vtl.from_array[f64](values, [groups, ptp_axes_width, ptp_axes_height])!
	for _ in 0 .. 2 {
		_ = stats.ptp_along_axes[f64](tensor, axes, false)!
	}
	mut samples := []f64{len: ptp_axes_iterations}
	mut checksum := f64(0)
	for i in 0 .. ptp_axes_iterations {
		started := time.sys_mono_now()
		result := stats.ptp_along_axes[f64](tensor, axes, false)!
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
		checksum = 0
		for value in result.to_array() {
			checksum += value
		}
	}
	mut total := f64(0)
	for sample in samples {
		total += sample
	}
	println('vtl ptp axes ${case_name} ${size} f64 | ${total / ptp_axes_iterations:.3} ms')
	println('checksum: ${int(checksum)}')
}
