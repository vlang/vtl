module main

import math.complex
import vtl
import vtl.fft

fn main() {
	signal := vtl.from_1d([1.0, 0.0, -1.0, 0.0])!
	println('Real FFT frequencies: ${fft.rfftfreq(signal.size, 0.25)!.to_array()}')

	complex_signal := vtl.from_1d[complex.Complex]([
		complex.complex(1, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
		complex.complex(0, 0),
	])!
	spectrum := fft.fft(complex_signal)!
	frequencies := fft.fftfreq(complex_signal.size, 0.25)!
	println('FFT frequencies: ${frequencies.to_array()}')
	println('Centered frequencies: ${fft.fftshift(frequencies)!.to_array()}')
	println('Centered spectrum: ${fft.fftshift(spectrum)!.to_array()}')
}
