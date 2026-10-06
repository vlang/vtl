module main

import vtl
import vtl.fft

fn main() {
	signal := vtl.from_1d([1.0, 0.0, -1.0, 0.0])!
	println('Real FFT frequencies: ${fft.rfftfreq(signal.size, 0.25)!.to_array()}')

	frequencies := fft.fftfreq(signal.size, 0.25)!
	println('FFT frequencies: ${frequencies.to_array()}')
	println('Centered frequencies: ${fft.fftshift(frequencies)!.to_array()}')
}
