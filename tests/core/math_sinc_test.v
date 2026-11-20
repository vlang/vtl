module core

import math
import vtl

fn test_sinc_f64_values_and_special_values() ! {
	input := vtl.from_1d[f64]([
		0.0,
		-0.0,
		1e-12,
		-1e-12,
		0.5,
		-0.5,
		1.0,
		-1.0,
		math.nan(),
		math.inf(1),
		math.inf(-1),
	])!
	result := input.sinc()!

	assert result.dtype() == .float64
	assert result.shape == input.shape
	assert result.to_array()[0] == 1.0
	assert result.to_array()[1] == 1.0
	assert math.abs(result.get_nth(2) - 1.0) < 1e-14
	assert math.abs(result.get_nth(3) - 1.0) < 1e-14
	assert math.abs(result.get_nth(4) - 2.0 / math.pi) < 1e-15
	assert math.abs(result.get_nth(5) - 2.0 / math.pi) < 1e-15
	assert math.abs(result.get_nth(6)) < 1e-15
	assert math.abs(result.get_nth(7)) < 1e-15
	assert math.is_nan(result.get_nth(8))
	assert math.is_nan(result.get_nth(9))
	assert math.is_nan(result.get_nth(10))
}

fn test_sinc_f32_preserves_dtype_and_handles_edge_values() ! {
	input := vtl.from_1d[f32]([
		0.0,
		-0.0,
		1e-5,
		-1e-5,
		0.5,
		-0.5,
		1.0,
		-1.0,
		f32(math.nan()),
		f32(math.inf(1)),
		f32(math.inf(-1)),
	])!
	result := input.sinc()!

	assert result.dtype() == .float32
	assert result.shape == input.shape
	assert result.get_nth(0) == f32(1.0)
	assert result.get_nth(1) == f32(1.0)
	assert math.abs(f64(result.get_nth(2)) - 1.0) < 1e-6
	assert math.abs(f64(result.get_nth(3)) - 1.0) < 1e-6
	assert math.abs(f64(result.get_nth(4)) - 2.0 / math.pi) < 1e-6
	assert math.abs(f64(result.get_nth(5)) - 2.0 / math.pi) < 1e-6
	assert math.abs(f64(result.get_nth(6))) < 1e-6
	assert math.abs(f64(result.get_nth(7))) < 1e-6
	assert math.is_nan(f64(result.get_nth(8)))
	assert math.is_nan(f64(result.get_nth(9)))
	assert math.is_nan(f64(result.get_nth(10)))
}

fn test_sinc_rejects_non_floating_tensors() ! {
	input := vtl.from_1d([1, 2, 3])!
	input.sinc() or {
		assert err.msg().contains('f32 or f64')
		return
	}
	assert false, 'expected sinc to reject integer tensors'
}

fn test_sinc_reads_noncontiguous_views() ! {
	input := vtl.from_1d[f64]([0.0, 0.5, 1.0, 1.5])!.flip()!
	result := input.sinc()!

	assert math.abs(result.get_nth(0) + 2.0 / (3.0 * math.pi)) < 1e-15
	assert math.abs(result.get_nth(1)) < 1e-15
	assert math.abs(result.get_nth(2) - 2.0 / math.pi) < 1e-15
	assert result.get_nth(3) == 1.0
}
