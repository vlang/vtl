// Compile with -prod from ~/.vmodules, then run the binary.
module main

import time
import vtl

const promoted_remainder_size = 1_048_576
const iterations = 7

fn main() {
	mut values := []i8{len: promoted_remainder_size}
	for i in 0 .. promoted_remainder_size {
		values[i] = i8(i % 251 - 125)
	}
	input := vtl.from_1d[i8](values) or { panic(err) }
	divisor := vtl.from_1d[i16]([17]) or { panic(err) }
	for _ in 0 .. 2 {
		_ = vtl.remainder_promoted[i16, i8, i16](input, divisor) or { panic(err) }
	}
	mut result := vtl.remainder_promoted[i16, i8, i16](input, divisor) or { panic(err) }
	mut samples := []f64{len: iterations}
	for i in 0 .. iterations {
		started := time.sys_mono_now()
		result = vtl.remainder_promoted[i16, i8, i16](input, divisor) or { panic(err) }
		samples[i] = f64(time.sys_mono_now() - started) / 1_000_000.0
	}
	mut checksum := i64(0)
	for value in result.data.data {
		checksum += i64(value)
	}
	mut total := f64(0)
	for sample in samples {
		total += sample
	}
	println('vtl remainder ${promoted_remainder_size} i8 % i16 scalar broadcast | ${total / iterations:.3} ms')
	println('checksum: ${checksum}')
}
