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
	mut plan := create_rfft_plan[f32](input.size)!
	assert plan.forward(input)!.array_equal(frequencies)
}

fn test_rfft_plan_reuses_backend_and_validates_input_length() ! {
	input := vtl.from_1d([f64(1), 0, -1, 0])!
	mut plan := create_rfft_plan[f64](4)!
	first := plan.forward(input)!
	second := plan.forward(input)!
	assert first.array_equal(second)

	wrong_length := vtl.from_1d([f64(1), 2])!
	_ := plan.forward(wrong_length) or {
		assert err.msg().contains('length 4')
		plan.destroy()
		return
	}
	assert false, 'expected reusable plan to reject a different input length'
}

fn test_rfft_plan_reuses_workspace_for_strided_inputs() ! {
	base := vtl.from_array([f64(1), 99, 0, 99, -1, 99, 0, 99], [4, 2])!
	view := base.slice([0, 2], []int{})!
	strided := view.as_strided([4], [2])!
	contiguous := vtl.from_1d([f64(1), 0, -1, 0])!
	mut plan := create_rfft_plan[f64](4)!
	before := strided.to_array()
	first := plan.forward(strided)!
	second := plan.forward(strided)!
	expected := rfft(contiguous)!
	assert first.array_equal(expected)
	assert second.array_equal(expected)
	assert strided.to_array() == before
	offset_base := vtl.from_1d([f64(99), 1, 0, -1, 0, 99])!
	offset_view := offset_base.slice([1, 5])!
	assert offset_view.is_row_major_contiguous()
	assert plan.forward(offset_view)!.array_equal(expected)
}

fn test_rfft_plan_rejects_use_after_destroy() ! {
	input := vtl.from_1d([f64(1), 0])!
	mut plan := create_rfft_plan[f64](2)!
	plan.destroy()
	plan.destroy()
	_ := plan.forward(input) or {
		assert err.msg().contains('destroyed')
		return
	}
	assert false, 'expected a destroyed plan to reject transforms'
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

fn test_complex_fft2_matches_expected_bins_and_roundtrips() ! {
	input := vtl.from_array[complex.Complex]([
		complex.complex(0, 0),
		complex.complex(1, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
	], [2, 2])!

	spectrum := fft2(input)!
	assert spectrum.shape == [2, 2]
	assert math.abs(spectrum.get([0, 0]).re - 1) < 1e-12
	assert math.abs(spectrum.get([0, 1]).re + 1) < 1e-12
	assert math.abs(spectrum.get([1, 0]).re - 1) < 1e-12
	assert math.abs(spectrum.get([1, 1]).re + 1) < 1e-12
	for i in 0 .. spectrum.size {
		assert math.abs(spectrum.get_nth(i).im) < 1e-12
	}

	reconstructed := ifft2(spectrum)!
	for i in 0 .. input.size {
		assert math.abs(reconstructed.get_nth(i).re - input.get_nth(i).re) < 1e-12
		assert math.abs(reconstructed.get_nth(i).im - input.get_nth(i).im) < 1e-12
	}

	general_spectrum := fftn(input)!
	assert general_spectrum.shape == spectrum.shape
	general_reconstructed := ifftn(general_spectrum)!
	for i in 0 .. input.size {
		assert math.abs(general_reconstructed.get_nth(i).re - input.get_nth(i).re) < 1e-12
	}
}

fn test_complex_fft2_handles_rectangular_tensors() ! {
	mut values := []complex.Complex{len: 6}
	values[3] = complex.complex(1, 0)
	input := vtl.from_array[complex.Complex](values, [2, 3])!
	spectrum := fft2(input)!
	assert spectrum.shape == [2, 3]
	for column in 0 .. 3 {
		assert math.abs(spectrum.get([0, column]).re - 1) < 1e-12
		assert math.abs(spectrum.get([1, column]).re + 1) < 1e-12
		assert math.abs(spectrum.get([0, column]).im) < 1e-12
		assert math.abs(spectrum.get([1, column]).im) < 1e-12
	}
}

fn test_complex_fft_axis_transforms_only_selected_axis_and_roundtrips() ! {
	input := vtl.from_array[complex.Complex]([
		complex.complex(0, 0),
		complex.complex(1, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
	], [2, 2])!
	columns := fft_axis(input, -1)!
	assert columns.shape == [2, 2]
	assert math.abs(columns.get([0, 0]).re - 1) < 1e-12
	assert math.abs(columns.get([0, 1]).re + 1) < 1e-12
	assert math.abs(columns.get([1, 0]).re) < 1e-12
	assert math.abs(columns.get([1, 1]).re) < 1e-12
	rows := fft_axis(input, 0)!
	assert rows.shape == input.shape
	assert math.abs(rows.get([0, 1]).re - 1) < 1e-12
	assert math.abs(rows.get([1, 1]).re - 1) < 1e-12
	reconstructed := ifft_axis(columns, 1)!
	for flat_index in 0 .. input.size {
		assert math.abs(reconstructed.get_nth(flat_index).re - input.get_nth(flat_index).re) < 1e-12
		assert math.abs(reconstructed.get_nth(flat_index).im - input.get_nth(flat_index).im) < 1e-12
	}
	if _ := fft_axis(input, 2) {
		assert false, 'fft_axis must reject an axis outside the tensor rank'
	} else {
		assert true
	}
}

fn test_fft2_rejects_wrong_rank() ! {
	input := vtl.from_1d[complex.Complex]([complex.complex(1, 0)])!
	_ := fft2(input) or {
		assert err.msg().contains('two-dimensional')
		return
	}
	assert false, 'expected fft2 to reject a vector'
}

fn test_rfft2_and_irfft2_roundtrip_even_and_odd_shapes() ! {
	input_even := vtl.from_array([f64(1), 0, 0, 0], [2, 2])!
	spectrum_even := rfft2[f64](input_even)!
	assert spectrum_even.shape == [2, 2]
	for i in 0 .. spectrum_even.size {
		assert math.abs(spectrum_even.get_nth(i).re - 1) < 1e-12
		assert math.abs(spectrum_even.get_nth(i).im) < 1e-12
	}
	reconstructed_even := irfft2(spectrum_even, [2, 2])!
	for i in 0 .. input_even.size {
		assert math.abs(reconstructed_even.get_nth(i) - input_even.get_nth(i)) < 1e-12
	}

	input_odd := vtl.from_array([f64(1), 2, 3, 4, 5, 6], [2, 3])!
	spectrum_odd := rfft2[f64](input_odd)!
	assert spectrum_odd.shape == [2, 2]
	reconstructed_odd := irfft2(spectrum_odd, [2, 3])!
	for i in 0 .. input_odd.size {
		assert math.abs(reconstructed_odd.get_nth(i) - input_odd.get_nth(i)) < 1e-12
	}

	impulse := vtl.from_array([f64(1), 0, 0, 0, 0, 0], [2, 3])!
	impulse_spectrum := rfft2[f64](impulse)!
	for i in 0 .. impulse_spectrum.size {
		assert math.abs(impulse_spectrum.get_nth(i).re - 1) < 1e-12
		assert math.abs(impulse_spectrum.get_nth(i).im) < 1e-12
	}
}

fn test_rfftn_and_irfftn_handle_three_dimensions() ! {
	input := vtl.from_array([f64(1), 2, 3, 4, 5, 6], [2, 1, 3])!
	spectrum := rfftn[f64](input)!
	assert spectrum.shape == [2, 1, 2]
	reconstructed := irfftn(spectrum, [2, 1, 3])!
	assert reconstructed.shape == input.shape
	for i in 0 .. input.size {
		assert math.abs(reconstructed.get_nth(i) - input.get_nth(i)) < 1e-12
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
