module stats

import vtl
import math

fn test_freq() {
	// Tests were also verified on Wolfram Alpha
	data := vtl.from_1d([10.0, 10.0, 5.9, 2.7])
	mut o := freq(data, 10.0)
	assert o == 2
	o = freq(data, 2.7)
	assert o == 1
	o = freq(data, 15)
	assert o == 0
}

fn tst_res(str1 string, str2 string) bool {
	if (math.abs(str1.f64() - str2.f64())) < 1e-5 {
		return true
	}
	return false
}

fn test_mean() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := mean(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '5.762500')
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = mean(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '17.650000')
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = mean(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '37.708000')
}

fn test_geometric_mean() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := geometric_mean(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '5.15993')
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = geometric_mean(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	ok := o.str() == 'nan' || o.str() == '-nan' || o.str() == '-1.#IND00' || o == f64(0)
		|| o.str() == '-nan(ind)'
	assert ok // Because in math it yields a complex number
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = geometric_mean(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '25.064496')
}

fn test_harmonic_mean() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := harmonic_mean(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '4.626519')
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = harmonic_mean(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '9.134577')
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = harmonic_mean(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '16.555477')
}

fn test_median() {
	// Tests were also verified on Wolfram Alpha
	// Assumes sorted array

	// Even
	mut data := vtl.from_1d([f64(2.7), f64(4.45), f64(5.9), f64(10.0)])
	mut o := median(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '5.175000')
	data = vtl.from_1d([f64(-3.0), f64(1.89), f64(4.4), f64(67.31)])
	o = median(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '3.145000')
	data = vtl.from_1d([f64(7.88), f64(12.0), f64(54.83), f64(76.122)])
	o = median(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '33.415000')

	// Odd
	data = vtl.from_1d([f64(2.7), f64(4.45), f64(5.9), f64(10.0), f64(22)])
	o = median(data)
	assert o == f64(5.9)
	data = vtl.from_1d([f64(-3.0), f64(1.89), f64(4.4), f64(9), f64(67.31)])
	o = median(data)
	assert o == f64(4.4)
	data = vtl.from_1d([f64(7.88), f64(3.3), f64(12.0), f64(54.83), f64(76.122)])
	o = median(data)
	assert o == f64(12.0)
}

fn test_mode() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(2.7), f64(2.7), f64(4.45), f64(5.9), f64(10.0)])
	mut o := mode(data)
	assert o == f64(2.7)
	data = vtl.from_1d([f64(-3.0), f64(1.89), f64(1.89), f64(1.89), f64(9), f64(4.4), f64(4.4),
		f64(9), f64(67.31)])
	o = mode(data)
	assert o == f64(1.89)
	// Testing greedy nature
	data = vtl.from_1d([f64(2.0), f64(4.0), f64(2.0), f64(4.0)])
	o = mode(data)
	assert o == f64(2.0)
}

fn test_rms() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := rms(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '6.362046')
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = rms(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '33.773393')
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = rms(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '47.452561')
}

fn test_population_variance() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := population_variance(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '7.269219')
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = population_variance(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '829.119550')
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = population_variance(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '829.852282')
}

fn test_sample_variance() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := sample_variance(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '9.692292')
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = sample_variance(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '1105.492733')
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = sample_variance(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '1106.469709')
}

fn test_population_stddev() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := population_stddev(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '2.696149')
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = population_stddev(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '28.794436')
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = population_stddev(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '28.807157')
}

fn test_sample_stddev() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := sample_stddev(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '3.113245')
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = sample_stddev(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '33.248951')
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = sample_stddev(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '33.263639')
}

fn test_absdev() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := absdev(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '2.187500')
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = absdev(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '24.830000')
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = absdev(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '27.768000')
}

fn test_min() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := min(data)
	assert o == f64(2.7)
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = min(data)
	assert o == f64(-3.0)
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = min(data)
	assert o == f64(7.88)
}

fn test_max() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := max(data)
	assert o == f64(10.0)
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = max(data)
	assert o == f64(67.31)
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = max(data)
	assert o == f64(76.122)
}

fn test_range() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := range(data)
	assert o == f64(7.3)
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = range(data)
	assert o == f64(70.31)
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = range(data)
	assert o == f64(68.242)
}

fn test_sum() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := sum(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '23.05')
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = sum(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '70.6')
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = sum(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '150.832')
}

fn test_prod() {
	// Tests were also verified on Wolfram Alpha
	mut data := vtl.from_1d([f64(10.0), f64(4.45), f64(5.9), f64(2.7)])
	mut o := prod(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '708.885')
	data = vtl.from_1d([f64(-3.0), f64(67.31), f64(4.4), f64(1.89)])
	o = prod(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '-1679.24988')
	data = vtl.from_1d([f64(12.0), f64(7.88), f64(76.122), f64(54.83)])
	o = prod(data)
	// Some issue with precision comparison in f64 using == operator hence serializing to string
	assert tst_res(o.str(), '394671.621226')
}

fn test_passing_empty() {
	data := vtl.from_1d([]f64{})
	assert freq(data, 0) == 0
	assert mean(data) == f64(0)
	assert geometric_mean(data) == f64(0)
	assert harmonic_mean(data) == f64(0)
	assert median(data) == f64(0)
	assert mode(data) == f64(0)
	assert rms(data) == f64(0)
	assert population_variance(data) == f64(0)
	assert sample_variance(data) == f64(0)
	assert population_stddev(data) == f64(0)
	assert sample_stddev(data) == f64(0)
	assert absdev(data) == f64(0)
	assert min(data) == f64(0)
	assert max(data) == f64(0)
	assert range(data) == f64(0)
	assert sum(data) == f64(0)
	assert prod(data) == f64(0)
}
