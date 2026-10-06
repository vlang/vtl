module fft

import math
import math.complex
import vtl

fn test_fftfreq_and_rfftfreq_match_numpy_frequency_order() ! {
	full := fftfreq(5, 1)!
	for i, expected in [0.0, 0.2, 0.4, -0.4, -0.2] {
		assert math.abs(full.get_nth(i) - expected) < 1e-12
	}
	half := rfftfreq(6, 2)!
	assert half.shape == [4]
	for i, expected in [0.0, 1.0 / 12.0, 1.0 / 6.0, 0.25] {
		assert math.abs(half.get_nth(i) - expected) < 1e-12
	}
}

fn test_fft_frequency_helpers_reject_invalid_lengths_and_spacing() {
	if _ := fftfreq(0, 1) {
		assert false, 'fftfreq must reject a zero transform length'
	}
	if _ := rfftfreq(-1, 1) {
		assert false, 'rfftfreq must reject a negative transform length'
	}
	if _ := fftfreq(4, 0) {
		assert false, 'frequency helpers must reject zero sample spacing'
	}
	if _ := rfftfreq(4, math.inf(1)) {
		assert false, 'frequency helpers must reject infinite sample spacing'
	}
}

fn test_fftshift_and_ifftshift_handle_odd_and_even_lengths() ! {
	odd := vtl.from_1d([0, 1, 2, 3, 4])!
	shifted := fftshift(odd)!
	assert shifted.to_array() == [3, 4, 0, 1, 2]
	assert ifftshift(odd)!.to_array() == [2, 3, 4, 0, 1]
	assert ifftshift(shifted)!.array_equal(odd)
	even := vtl.from_1d([0, 1, 2, 3])!
	assert fftshift(even)!.to_array() == [2, 3, 0, 1]
	assert ifftshift(even)!.to_array() == [2, 3, 0, 1]
}

fn test_fftshift_supports_complex_fft_output() ! {
	input := vtl.from_1d[complex.Complex]([
		complex.complex(1, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
	])!
	shifted := fftshift(fft(input)!)!
	assert shifted.shape == [4]
	for index in 0 .. shifted.size {
		assert shifted.get_nth(index).re == 1
		assert shifted.get_nth(index).im == 0
	}
	assert ifftshift(shifted)!.array_equal(fft(input)!)
}

fn test_fftshift_axis_handles_negative_axis_and_strided_input() ! {
	input := vtl.from_2d([[0, 1, 2], [3, 4, 5]])!.transpose([1, 0])!
	shifted := fftshift_axis(input, -1)!
	assert shifted.shape == [3, 2]
	assert shifted.to_array() == [3, 0, 4, 1, 5, 2]
	restored := ifftshift_axis(shifted, 1)!
	assert restored.array_equal(input)
	shifted_all := fftshift(input)!
	assert shifted_all.to_array() == [5, 2, 3, 0, 4, 1]
	assert ifftshift(shifted_all)!.array_equal(input)
}

fn test_fftshift_axis_rejects_invalid_axis() {
	scalar := vtl.tensor(1, [])
	if _ := fftshift_axis(scalar, 0) {
		assert false, 'fftshift_axis must reject an invalid axis'
	}
}
