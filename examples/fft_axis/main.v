module main

import math
import math.complex
import vtl
import vtl.fft

fn main() {
	image := vtl.from_array[complex.Complex]([
		complex.complex(0, 0),
		complex.complex(1, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
	], [2, 2])!
	column_spectrum := fft.fft_axis(image, -1)!
	reconstructed := fft.ifft_axis(column_spectrum, 1)!
	assert math.abs(column_spectrum.get([0, 0]).re - 1) < 1e-12
	assert math.abs(column_spectrum.get([0, 1]).re + 1) < 1e-12
	for flat_index in 0 .. image.size {
		assert math.abs(reconstructed.get_nth(flat_index).re - image.get_nth(flat_index).re) < 1e-12
	}
	real_image := vtl.from_array([f64(0), 1, 0, 0], [2, 2])!
	real_spectrum := fft.rfft_axis(real_image, 1)!
	real_reconstructed := fft.irfft_axis(real_spectrum, 1, real_image.shape[1])!
	for flat_index in 0 .. real_image.size {
		assert math.abs(real_reconstructed.get_nth(flat_index) - real_image.get_nth(flat_index)) < 1e-12
	}
	println('Input shape: ${image.shape}')
	println('FFT along the final axis: ${column_spectrum.to_array()}')
	println('Inverse along axis 1: ${reconstructed.to_array()}')
	println('Real FFT shape along axis 1: ${real_spectrum.shape}')
}
