module main

import math
import vtl
import vtl.fft

fn test_complex_f32_fft_round_trip_and_numpy_forward_order() ! {
	input := vtl.from_1d[fft.Complex32]([
		fft.Complex32{ re: 1, im: 0 },
		fft.Complex32{ re: 0, im: 0 },
	])!
	spectrum := fft.fft_f32(input)!
	assert spectrum.shape == [2]
	assert math.abs(spectrum.get_nth(0).re - 1) < 1e-5
	assert math.abs(spectrum.get_nth(1).re - 1) < 1e-5
	restored := fft.ifft_f32(spectrum)!
	assert math.abs(restored.get_nth(0).re - 1) < 1e-5
	assert math.abs(restored.get_nth(1).re) < 1e-5
}

fn test_complex_f32_fft_preserves_imaginary_components() ! {
	input := vtl.from_1d[fft.Complex32]([
		fft.Complex32{ re: 1, im: 2 },
		fft.Complex32{ re: 3, im: 4 },
	])!
	spectrum := fft.fft_f32(input)!
	assert math.abs(spectrum.get_nth(0).re - 4) < 1e-5
	assert math.abs(spectrum.get_nth(0).im - 6) < 1e-5
	assert math.abs(spectrum.get_nth(1).re + 2) < 1e-5
	assert math.abs(spectrum.get_nth(1).im + 2) < 1e-5
	restored := fft.ifft_f32(spectrum)!
	for i in 0 .. input.size {
		assert math.abs(restored.get_nth(i).re - input.get_nth(i).re) < 1e-5
		assert math.abs(restored.get_nth(i).im - input.get_nth(i).im) < 1e-5
	}
}

fn test_complex_f32_fft_reads_strided_input_in_logical_order() ! {
	base := vtl.from_1d[fft.Complex32]([
		fft.Complex32{ re: 1, im: 2 },
		fft.Complex32{ re: 99 },
		fft.Complex32{ re: 3, im: 4 },
		fft.Complex32{ re: 99 },
	])!
	strided := base.slice([0, 4, 2])!
	contiguous := vtl.from_1d[fft.Complex32](strided.to_array())!
	assert !strided.is_row_major_contiguous()
	strided_spectrum := fft.fft_f32(strided)!
	contiguous_spectrum := fft.fft_f32(contiguous)!
	for i in 0 .. strided_spectrum.size {
		assert math.abs(strided_spectrum.get_nth(i).re - contiguous_spectrum.get_nth(i).re) < 1e-5
		assert math.abs(strided_spectrum.get_nth(i).im - contiguous_spectrum.get_nth(i).im) < 1e-5
	}
}

fn test_complex_f32_axis_and_nd_transforms_preserve_shape() ! {
	input := vtl.from_array[fft.Complex32]([
		fft.Complex32{ re: 1, im: 0 },
		fft.Complex32{ re: 0, im: 0 },
		fft.Complex32{ re: 0, im: 0 },
		fft.Complex32{ re: 0, im: 0 },
	], [2, 2])!
	axis_spectrum := fft.fft_axis_f32(input, -1)!
	assert axis_spectrum.shape == [2, 2]
	axis_restored := fft.ifft_axis_f32(axis_spectrum, 1)!
	nd_spectrum := fft.fftn_f32(input)!
	assert nd_spectrum.shape == [2, 2]
	nd_restored := fft.ifftn_f32(nd_spectrum)!
	for i in 0 .. input.size {
		assert math.abs(axis_restored.get_nth(i).re - input.get_nth(i).re) < 1e-5
		assert math.abs(nd_restored.get_nth(i).re - input.get_nth(i).re) < 1e-5
	}
	two_d_spectrum := fft.fft2_f32(input)!
	two_d_restored := fft.ifft2_f32(two_d_spectrum)!
	assert two_d_spectrum.shape == [2, 2]
	for i in 0 .. input.size {
		assert math.abs(two_d_restored.get_nth(i).re - input.get_nth(i).re) < 1e-5
	}
}

fn test_complex_f32_axis_and_nd_transforms_read_strided_views() ! {
	base := vtl.from_array[fft.Complex32]([
		fft.Complex32{ re: 1 },
		fft.Complex32{ re: 99 },
		fft.Complex32{ re: 2, im: 1 },
		fft.Complex32{ re: 99 },
		fft.Complex32{ re: 3, im: -1 },
		fft.Complex32{ re: 99 },
		fft.Complex32{ re: 4, im: 2 },
		fft.Complex32{ re: 99 },
		fft.Complex32{ re: 5, im: 1 },
		fft.Complex32{ re: 99 },
		fft.Complex32{ re: 6, im: -2 },
		fft.Complex32{ re: 99 },
		fft.Complex32{ re: 7, im: 3 },
		fft.Complex32{ re: 99 },
		fft.Complex32{ re: 8 },
		fft.Complex32{ re: 99 },
	], [2, 8])!
	strided := base.slice([]int{}, [0, 8, 2])!
	contiguous := vtl.from_array[fft.Complex32](strided.to_array(), strided.shape)!
	assert !strided.is_row_major_contiguous()
	axis_result := fft.fft_axis_f32(strided, 1)!
	axis_expected := fft.fft_axis_f32(contiguous, 1)!
	for i in 0 .. axis_result.size {
		assert math.abs(axis_result.get_nth(i).re - axis_expected.get_nth(i).re) < 1e-5
		assert math.abs(axis_result.get_nth(i).im - axis_expected.get_nth(i).im) < 1e-5
	}
	nd_result := fft.fftn_f32(strided)!
	nd_expected := fft.fftn_f32(contiguous)!
	for i in 0 .. nd_result.size {
		assert math.abs(nd_result.get_nth(i).re - nd_expected.get_nth(i).re) < 1e-5
		assert math.abs(nd_result.get_nth(i).im - nd_expected.get_nth(i).im) < 1e-5
	}
}

fn test_complex_f32_fft_normalization_modes() ! {
	input := vtl.from_1d[fft.Complex32]([
		fft.Complex32{ re: 1, im: 0 },
		fft.Complex32{ re: 0, im: 0 },
		fft.Complex32{ re: 0, im: 0 },
		fft.Complex32{ re: 0, im: 0 },
	])!
	backward := fft.fft_norm_f32(input, .backward)!
	forward := fft.fft_norm_f32(input, .forward)!
	ortho := fft.fft_norm_f32(input, .ortho)!
	for i in 0 .. input.size {
		assert math.abs(backward.get_nth(i).re - 1) < 1e-5
		assert math.abs(forward.get_nth(i).re - 0.25) < 1e-5
		assert math.abs(ortho.get_nth(i).re - 0.5) < 1e-5
	}
	restored := fft.ifft_norm_f32(forward, .forward)!
	assert math.abs(restored.get_nth(0).re - 1) < 1e-5
	for i in 1 .. input.size {
		assert math.abs(restored.get_nth(i).re) < 1e-5
	}
}
