module fft

import math
import vtl

fn test_complex_f32_fft_round_trip_and_numpy_forward_order() ! {
	input := vtl.from_1d[Complex32]([
		Complex32{ re: 1, im: 0 },
		Complex32{ re: 0, im: 0 },
	])!
	spectrum := fft_f32(input)!
	assert spectrum.shape == [2]
	assert math.abs(spectrum.get_nth(0).re - 1) < 1e-5
	assert math.abs(spectrum.get_nth(1).re - 1) < 1e-5
	restored := ifft_f32(spectrum)!
	assert math.abs(restored.get_nth(0).re - 1) < 1e-5
	assert math.abs(restored.get_nth(1).re) < 1e-5
}

fn test_complex_f32_fft_preserves_imaginary_components() ! {
	input := vtl.from_1d[Complex32]([
		Complex32{ re: 1, im: 2 },
		Complex32{ re: 3, im: 4 },
	])!
	spectrum := fft_f32(input)!
	assert math.abs(spectrum.get_nth(0).re - 4) < 1e-5
	assert math.abs(spectrum.get_nth(0).im - 6) < 1e-5
	assert math.abs(spectrum.get_nth(1).re + 2) < 1e-5
	assert math.abs(spectrum.get_nth(1).im + 2) < 1e-5
	restored := ifft_f32(spectrum)!
	for i in 0 .. input.size {
		assert math.abs(restored.get_nth(i).re - input.get_nth(i).re) < 1e-5
		assert math.abs(restored.get_nth(i).im - input.get_nth(i).im) < 1e-5
	}
}

fn test_complex_f32_axis_and_nd_transforms_preserve_shape() ! {
	input := vtl.from_array[Complex32]([
		Complex32{ re: 1, im: 0 },
		Complex32{ re: 0, im: 0 },
		Complex32{ re: 0, im: 0 },
		Complex32{ re: 0, im: 0 },
	], [2, 2])!
	axis_spectrum := fft_axis_f32(input, -1)!
	assert axis_spectrum.shape == [2, 2]
	axis_restored := ifft_axis_f32(axis_spectrum, 1)!
	nd_spectrum := fftn_f32(input)!
	assert nd_spectrum.shape == [2, 2]
	nd_restored := ifftn_f32(nd_spectrum)!
	for i in 0 .. input.size {
		assert math.abs(axis_restored.get_nth(i).re - input.get_nth(i).re) < 1e-5
		assert math.abs(nd_restored.get_nth(i).re - input.get_nth(i).re) < 1e-5
	}
	two_d_spectrum := fft2_f32(input)!
	two_d_restored := ifft2_f32(two_d_spectrum)!
	assert two_d_spectrum.shape == [2, 2]
	for i in 0 .. input.size {
		assert math.abs(two_d_restored.get_nth(i).re - input.get_nth(i).re) < 1e-5
	}
}

fn test_complex_f32_fft_normalization_modes() ! {
	input := vtl.from_1d[Complex32]([
		Complex32{ re: 1, im: 0 },
		Complex32{ re: 0, im: 0 },
		Complex32{ re: 0, im: 0 },
		Complex32{ re: 0, im: 0 },
	])!
	backward := fft_norm_f32(input, .backward)!
	forward := fft_norm_f32(input, .forward)!
	ortho := fft_norm_f32(input, .ortho)!
	for i in 0 .. input.size {
		assert math.abs(backward.get_nth(i).re - 1) < 1e-5
		assert math.abs(forward.get_nth(i).re - 0.25) < 1e-5
		assert math.abs(ortho.get_nth(i).re - 0.5) < 1e-5
	}
	restored := ifft_norm_f32(forward, .forward)!
	assert math.abs(restored.get_nth(0).re - 1) < 1e-5
	for i in 1 .. input.size {
		assert math.abs(restored.get_nth(i).re) < 1e-5
	}
}
