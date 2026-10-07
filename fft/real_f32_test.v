module fft

import math
import vtl

fn test_real_f32_fft_has_complex_f32_bins_and_round_trips() ! {
	input := vtl.from_1d[f32]([1, 0, 0, 0])!
	spectrum := rfft_f32(input)!
	assert spectrum.shape == [3]
	for i in 0 .. spectrum.size {
		assert math.abs(spectrum.get_nth(i).re - 1) < 1e-5
		assert math.abs(spectrum.get_nth(i).im) < 1e-5
	}
	restored := irfft_f32(spectrum, input.size)!
	for i in 0 .. input.size {
		assert math.abs(restored.get_nth(i) - input.get_nth(i)) < 1e-5
	}
}

fn test_real_f32_fft_round_trips_odd_lengths() ! {
	input := vtl.from_1d[f32]([1, 2, 3])!
	spectrum := rfft_f32(input)!
	assert spectrum.shape == [2]
	assert math.abs(spectrum.get_nth(0).re - 6) < 1e-5
	assert math.abs(spectrum.get_nth(1).re + 1.5) < 1e-5
	assert math.abs(spectrum.get_nth(1).im - 0.8660254) < 1e-5
	restored := irfft_f32(spectrum, input.size)!
	for i in 0 .. input.size {
		assert math.abs(restored.get_nth(i) - input.get_nth(i)) < 1e-5
	}
}

fn test_reusable_real_f32_fft_plan() ! {
	input := vtl.from_1d[f32]([1, 0, 0, 0])!
	mut plan := create_rfft_f32_plan(input.size)!
	defer {
		plan.destroy()
	}
	first := plan.forward(input)!
	second := plan.forward(input)!
	assert first.shape == [3]
	assert second.shape == first.shape
	assert math.abs(second.get_nth(0).re - 1) < 1e-5
	assert math.abs(second.get_nth(1).im) < 1e-5
}

fn test_real_f32_axis_and_nd_transforms_round_trip() ! {
	input := vtl.from_array[f32]([1, 0, 0, 0], [2, 2])!
	axis_spectrum := rfft_axis_f32(input, -1)!
	assert axis_spectrum.shape == [2, 2]
	axis_restored := irfft_axis_f32(axis_spectrum, 1, 2)!
	nd_spectrum := rfftn_f32(input)!
	assert nd_spectrum.shape == [2, 2]
	nd_restored := irfftn_f32(nd_spectrum, input.shape)!
	two_d_spectrum := rfft2_f32(input)!
	two_d_restored := irfft2_f32(two_d_spectrum, input.shape)!
	for i in 0 .. input.size {
		assert math.abs(axis_restored.get_nth(i) - input.get_nth(i)) < 1e-5
		assert math.abs(nd_restored.get_nth(i) - input.get_nth(i)) < 1e-5
		assert math.abs(two_d_restored.get_nth(i) - input.get_nth(i)) < 1e-5
	}
}

fn test_real_f32_fft_normalization_matches_numpy_modes() ! {
	input := vtl.from_1d[f32]([1, 0, 0, 0])!
	backward := rfft_norm_f32(input, .backward)!
	forward := rfft_norm_f32(input, .forward)!
	ortho := rfft_norm_f32(input, .ortho)!
	for i in 0 .. backward.size {
		assert math.abs(backward.get_nth(i).re - 1) < 1e-5
		assert math.abs(forward.get_nth(i).re - 0.25) < 1e-5
		assert math.abs(ortho.get_nth(i).re - 0.5) < 1e-5
	}
	restored := irfft_norm_f32(forward, input.size, .forward)!
	for i in 0 .. input.size {
		assert math.abs(restored.get_nth(i) - input.get_nth(i)) < 1e-5
	}
	assert input.get_nth(0) == 1
}

fn test_real_f32_axis_and_nd_normalization_round_trip() ! {
	input := vtl.from_array[f32]([1, 0, 0, 0], [2, 2])!
	axis_forward := rfft_axis_norm_f32(input, 1, .forward)!
	axis_restored := irfft_axis_norm_f32(axis_forward, 1, 2, .forward)!
	nd_forward := rfftn_norm_f32(input, .forward)!
	nd_restored := irfftn_norm_f32(nd_forward, input.shape, .forward)!
	two_d_forward := rfft2_norm_f32(input, .ortho)!
	two_d_restored := irfft2_norm_f32(two_d_forward, input.shape, .ortho)!
	for i in 0 .. input.size {
		assert math.abs(axis_restored.get_nth(i) - input.get_nth(i)) < 1e-5
		assert math.abs(nd_restored.get_nth(i) - input.get_nth(i)) < 1e-5
		assert math.abs(two_d_restored.get_nth(i) - input.get_nth(i)) < 1e-5
	}
}

fn test_real_f32_fft_rejects_incompatible_shapes() {
	input := vtl.from_1d[f32]([1, 2, 3, 4])!
	if _ := rfft_f32(vtl.from_array[f32]([1, 2], [1, 2])!) {
		assert false
	}
	spectrum := rfft_f32(input)!
	if _ := irfft_f32(spectrum, 6) {
		assert false
	}
	if _ := rfft2_f32(input) {
		assert false
	}
}
