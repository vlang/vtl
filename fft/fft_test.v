module fft

import math
import math.complex
import vtl

fn test_rfft_decodes_even_length_frequencies_and_irfft_roundtrips() ! {
	input := vtl.from_1d([f64(0.5), 0.5, 1, 2])!

	frequencies := rfft[f64](input)!
	assert frequencies.shape == [3]
	assert math.abs(frequencies.get_nth(0).re - 4) < 1e-12
	assert math.abs(frequencies.get_nth(0).im) < 1e-12
	assert math.abs(frequencies.get_nth(1).re + 0.5) < 1e-12
	assert math.abs(frequencies.get_nth(1).im - 1.5) < 1e-12
	assert math.abs(frequencies.get_nth(2).re + 1) < 1e-12
	assert math.abs(frequencies.get_nth(2).im) < 1e-12

	reconstructed := irfft(frequencies, 4)!
	for i in 0 .. input.size {
		assert math.abs(reconstructed.get_nth(i) - input.get_nth(i)) < 1e-12
	}
}

fn test_rfft_and_irfft_handle_odd_lengths() ! {
	input := vtl.from_1d([f64(1), 2, 3])!

	frequencies := rfft[f64](input)!
	assert frequencies.shape == [2]
	assert math.abs(frequencies.get_nth(0).re - 6) < 1e-12
	assert math.abs(frequencies.get_nth(1).re + 1.5) < 1e-12
	assert math.abs(frequencies.get_nth(1).im - math.sqrt(3) / 2) < 1e-12

	reconstructed := irfft(frequencies, 3)!
	for i in 0 .. input.size {
		assert math.abs(reconstructed.get_nth(i) - input.get_nth(i)) < 1e-12
	}
}

fn test_rfft_accepts_f32_and_returns_f64_complex_values() ! {
	input := vtl.from_1d([f32(1), 0, -1, 0])!

	frequencies := rfft[f32](input)!
	assert frequencies.shape == [3]
	assert math.abs(frequencies.get_nth(0).re) < 1e-6
	assert math.abs(frequencies.get_nth(1).re - 2) < 1e-6
	assert math.abs(frequencies.get_nth(2).re) < 1e-6
}

fn test_complex_fft_and_ifft_roundtrip() ! {
	input := vtl.from_1d[complex.Complex]([
		complex.complex(1, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
	])!

	spectrum := fft(input)!
	assert spectrum.shape == [4]
	for i in 0 .. input.size {
		assert math.abs(spectrum.get_nth(i).re - 1) < 1e-12
		assert math.abs(spectrum.get_nth(i).im) < 1e-12
	}

	phase_input := vtl.from_1d[complex.Complex]([
		complex.complex(0, 0),
		complex.complex(1, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
	])!
	phase_spectrum := fft(phase_input)!
	assert math.abs(phase_spectrum.get_nth(0).re - 1) < 1e-12
	assert math.abs(phase_spectrum.get_nth(1).im + 1) < 1e-12
	assert math.abs(phase_spectrum.get_nth(2).re + 1) < 1e-12
	assert math.abs(phase_spectrum.get_nth(3).im - 1) < 1e-12

	reconstructed := ifft(spectrum)!
	for i in 0 .. input.size {
		assert math.abs(reconstructed.get_nth(i).re - input.get_nth(i).re) < 1e-12
		assert math.abs(reconstructed.get_nth(i).im - input.get_nth(i).im) < 1e-12
	}
}

fn test_rfft_rejects_non_vector_and_empty_input() ! {
	matrix := vtl.ones[f64]([2, 2])

	_ := rfft[f64](matrix) or {
		assert err.msg().contains('one-dimensional')
		return
	}
	assert false, 'expected non-vector input to fail'
}

fn test_rfft_rejects_empty_input() ! {
	empty := vtl.zeros[f64]([0])

	_ := rfft[f64](empty) or {
		assert err.msg().contains('at least one')
		return
	}
	assert false, 'expected empty input to fail'
}

fn test_rfft_rejects_integer_input() ! {
	input := vtl.from_1d([i64(1), 2])!

	_ := rfft[i64](input) or {
		assert err.msg().contains('f32 and f64')
		return
	}
	assert false, 'expected integer input to fail'
}

fn test_irfft_rejects_invalid_length_and_bin_count() ! {
	frequencies := vtl.from_1d[complex.Complex]([complex.complex(1, 0), complex.complex(2, 0)])!

	_ := irfft(frequencies, 4) or {
		assert err.msg().contains('frequency bins')
		return
	}
	assert false, 'expected invalid bin count to fail'
}
