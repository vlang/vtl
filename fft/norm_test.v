module fft

import math
import math.complex
import vtl

fn test_complex_fft_normalization_modes_match_numpy_conventions() ! {
	impulse := vtl.from_1d[complex.Complex]([
		complex.complex(1, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
	])!

	backward := fft_norm(impulse, .backward)!
	forward := fft_norm(impulse, .forward)!
	ortho := fft_norm(impulse, .ortho)!
	for i in 0 .. impulse.size {
		assert math.abs(backward.get_nth(i).re - 1) < 1e-12
		assert math.abs(forward.get_nth(i).re - 0.25) < 1e-12
		assert math.abs(ortho.get_nth(i).re - 0.5) < 1e-12
		assert math.abs(impulse.get_nth(i).re - (if i == 0 { 1 } else { 0 })) < 1e-12
	}

	mut spectrum_values := []complex.Complex{len: 4, init: complex.complex(1, 0)}
	spectrum := vtl.from_1d[complex.Complex](spectrum_values)!
	backward_inverse := ifft_norm(spectrum, .backward)!
	forward_inverse := ifft_norm(spectrum, .forward)!
	ortho_inverse := ifft_norm(spectrum, .ortho)!
	assert math.abs(backward_inverse.get_nth(0).re - 1) < 1e-12
	assert math.abs(forward_inverse.get_nth(0).re - 4) < 1e-12
	assert math.abs(ortho_inverse.get_nth(0).re - 2) < 1e-12
	for i in 1 .. impulse.size {
		assert math.abs(backward_inverse.get_nth(i).re) < 1e-12
		assert math.abs(forward_inverse.get_nth(i).re) < 1e-12
		assert math.abs(ortho_inverse.get_nth(i).re) < 1e-12
	}
	assert math.abs(spectrum.get_nth(0).re - 1) < 1e-12
}

fn test_axis_and_nd_fft_normalization_use_transformed_size() ! {
	impulse := vtl.from_array[complex.Complex]([
		complex.complex(1, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
	], [2, 2])!
	axis := fft_axis_norm(impulse, 0, .forward)!
	assert axis.shape == [2, 2]
	assert math.abs(axis.get_nth(0).re - 0.5) < 1e-12
	assert math.abs(impulse.get_nth(0).re - 1) < 1e-12
	nd := fftn_norm(impulse, .forward)!
	assert math.abs(nd.get_nth(0).re - 0.25) < 1e-12
	nd_inverse := ifftn_norm(nd, .forward)!
	for i in 0 .. impulse.size {
		assert math.abs(nd_inverse.get_nth(i).re - impulse.get_nth(i).re) < 1e-12
	}
}

fn test_real_fft_normalization_modes_roundtrip() ! {
	impulse := vtl.from_1d([f64(1), 0, 0, 0])!
	forward := rfft_norm[f64](impulse, .forward)!
	assert forward.shape == [3]
	for i in 0 .. forward.size {
		assert math.abs(forward.get_nth(i).re - 0.25) < 1e-12
	}
	restored := irfft_norm(forward, 4, .forward)!
	for i in 0 .. impulse.size {
		assert math.abs(restored.get_nth(i) - impulse.get_nth(i)) < 1e-12
		assert math.abs(impulse.get_nth(i) - (if i == 0 { 1 } else { 0 })) < 1e-12
	}
	nd := rfftn_norm[f64](vtl.from_array([f64(1), 0, 0, 0], [2, 2])!, .ortho)!
	assert nd.shape == [2, 2]
	assert math.abs(nd.get_nth(0).re - 0.5) < 1e-12
}
