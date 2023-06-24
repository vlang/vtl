module vtl

fn test_abs() {
	a := from_1d([-1, 2, -3, 4])!
	result := a.abs()
	expected := from_1d([1, 2, 3, 4])!
	assert result.array_equal(expected)
}

fn test_acos() {
	a := from_1d([-1.0, 0, 1])!
	result := a.acos()
	expected := from_1d([3.141592653589793, 1.5707963267948966, 0])!
	assert result.array_equal(expected)
}

fn test_acosh() {
	a := from_1d([1.0, 2, 3])!
	result := a.acosh()
	expected := from_1d([0.0, 1.3169578969248166, 1.7627471740390859])!
	assert result.array_equal(expected)
}

fn test_asin() {
	a := from_1d([-1.0, 0, 1])!
	result := a.asin()
	expected := from_1d([-1.5707963267948966, 0.0, 1.5707963267948966])!
	assert result.array_equal(expected)
}

fn test_asinh() {
	a := from_1d([-1.0, 0, 1])!
	result := a.asinh()
	expected := from_1d([-0.881373587019543, 0.0, 0.881373587019543])!
	assert result.array_equal(expected)
}

fn test_atan() {
	a := from_1d([-1.0, 0, 1])!
	result := a.atan()
	expected := from_1d([-0.7853981633974483, 0.0, 0.7853981633974483])!
	assert result.array_equal(expected)
}

fn test_atan2() {
	a := from_1d([1.0, 2, 3])!
	b := from_1d([4.0, 5, 6])!
	result := a.atan2(b)!
	expected := from_1d([0.24497866312686414, 0.3805063771123649, 0.4636476090008061])!
	assert result.array_equal(expected)
}

fn test_atanh() {
	a := from_1d([-0.5, 0, 0.5])!
	result := a.atanh()
	expected := from_1d([-0.5493061443340548, 0.0, 0.5493061443340548])!
	assert result.array_equal(expected)
}

fn test_cbrt() {
	a := from_1d([1.0, 2, 3])!
	result := a.cbrt()
	expected := from_1d([1.0, 1.2599210498948732, 1.4422495703074083])!
	assert result.array_equal(expected)
}

fn test_ceil() {
	a := from_1d([1.0, 2, 3])!
	result := a.ceil()
	expected := from_1d([1.0, 2.0, 3.0])!
	assert result.array_equal(expected)
}

fn test_cos() {
	a := from_1d([1.0, 2, 3])!
	result := a.cos()
	expected := from_1d([0.5403023058681398, -0.4161468365471424, -0.9899924966004454])!
	assert result.array_equal(expected)
}

fn test_cosh() {
	a := from_1d([1.0, 2, 3])!
	result := a.cosh()
	expected := from_1d([1.5430806348152437, 3.7621956910836314, 10.067661995777765])!
	assert result.array_equal(expected)
}

fn test_cot() {
	a := from_1d([1.0, 2, 3])!
	result := a.cot()
	expected := from_1d([0.6420926159343308, -0.45765755436028577, -7.015252551434534])!
	assert result.array_equal(expected)
}

fn test_degrees() {
	a := from_1d([1.0, 2, 3])!
	result := a.degrees()
	expected := from_1d([57.29577951308232, 114.59155902616465, 171.88733853924697])!
	assert result.array_equal(expected)
}

fn test_erf() {
	a := from_1d([1.0, 2, 3])!
	result := a.erf()
	expected := from_1d([0.8427007929497149, 0.9953222650189527, 0.9999779095030014])!
	assert result.array_equal(expected)
}

fn test_erfc() {
	a := from_1d([1.0, 2, 3])!
	result := a.erfc()
	expected := from_1d([0.15729920705028513, 0.004677734981047265, 2.2090496998585438e-05])!
	assert result.array_equal(expected)
}

fn test_exp() {
	a := from_1d([1.0, 2, 3])!
	result := a.exp()
	expected := from_1d([2.718281828459045, 7.38905609893065, 20.085536923187668])!
	assert result.array_equal(expected)
}

fn test_exp2() {
	a := from_1d([1.0, 2, 3])!
	result := a.exp2()
	expected := from_1d([2.0, 4.0, 8.0])!
	assert result.array_equal(expected)
}

fn test_expm1() {
	a := from_1d([1.0, 2, 3])!
	result := a.expm1()
	expected := from_1d([1.718281828459045, 6.38905609893065, 19.085536923187668])!
	assert result.array_equal(expected)
}

fn test_factorial() {
	a := from_1d([1.0, 2, 3])!
	result := a.factorial()
	expected := from_1d([1.0, 2.0, 6.0])!
	assert result.array_equal(expected)
}

fn test_floor() {
	a := from_1d([1.0, 2, 3])!
	result := a.floor()
	expected := from_1d([1.0, 2.0, 3.0])!
	assert result.array_equal(expected)
}

fn test_fmod() {
	a := from_1d([1.0, 2, 3])!
	b := from_1d([4.0, 5, 6])!
	result := a.fmod(b)!
	expected := from_1d([1.0, 2.0, 3.0])!
	assert result.array_equal(expected)
}

fn test_gamma() {
	a := from_1d([1.0, 2, 3])!
	result := a.gamma()
	expected := from_1d([1.0, 1.0, 2.0])!
	assert result.array_equal(expected)
}

fn test_gcd() {
	a := from_1d([1.0, 2, 3])!
	b := from_1d([4.0, 5, 6])!
	result := a.gcd(b)!
	expected := from_1d([1.0, 1.0, 3.0])!
	assert result.array_equal(expected)
}

fn test_hypot() {
	a := from_1d([1.0, 2, 3])!
	b := from_1d([4.0, 5, 6])!
	result := a.hypot(b)!
	expected := from_1d([4.123105625617661, 5.385164807134504, 6.708203932499369])!
	assert result.array_equal(expected)
}

fn test_lcm() {
	a := from_1d([1.0, 2, 3])!
	b := from_1d([4.0, 5, 6])!
	result := a.lcm(b)!
	expected := from_1d([4.0, 10.0, 6.0])!
	assert result.array_equal(expected)
}

fn test_log() {
	a := from_1d([1.0, 2, 3])!
	result := a.log()
	expected := from_1d([0.0, 0.6931471805599453, 1.0986122886681096])!
	assert result.array_equal(expected)
}

fn test_log10() {
	a := from_1d([1.0, 2, 3])!
	result := a.log10()
	expected := from_1d([0.0, 0.30102999566398114, 0.4771212547196623])!
	assert result.array_equal(expected)
}

fn test_log1p() {
	a := from_1d([1.0, 2, 3])!
	result := a.log1p()
	expected := from_1d([0.6931471805599453, 1.0986122886681096, 1.3862943611198906])!
	assert result.array_equal(expected)
}

fn test_log2() {
	a := from_1d([1.0, 2, 3])!
	result := a.log2()
	expected := from_1d([0.0, 1.0, 1.5849625007211563])!
	assert result.array_equal(expected)
}

fn test_log_factorial() {
	a := from_1d([1.0, 2, 3])!
	result := a.log_factorial()
	expected := from_1d([0.0, 0.6931471805599453, 1.791759469228055])!
	assert result.array_equal(expected)
}

fn test_log_gamma() {
	a := from_1d([1.0, 2, 3])!
	result := a.log_gamma()
	expected := from_1d([0.0, 0.0, 0.6931471805599453])!
	assert result.array_equal(expected)
}

fn test_log_n() {
	a := from_1d([1.0, 2, 3])!
	b := from_1d([4.0, 5, 6])!
	result := a.log_n(b)!
	expected := from_1d([0.0, 0.43067655807339306, 0.6131471927654584])!
	assert result.array_equal(expected)
}

fn test_max() {
	a := from_1d([1.0, 2, 3])!
	b := from_1d([4.0, 5, 6])!
	result := a.max(b)!
	expected := from_1d([4.0, 5.0, 6.0])!
	assert result.array_equal(expected)
}

fn test_min() {
	a := from_1d([1.0, 2, 3])!
	b := from_1d([4.0, 5, 6])!
	result := a.min(b)!
	expected := from_1d([1.0, 2.0, 3.0])!
	assert result.array_equal(expected)
}

fn test_pow() {
	a := from_1d([1.0, 2, 3])!
	b := from_1d([4.0, 5, 6])!
	result := a.pow(b)!
	expected := from_1d([1.0, 32.0, 729.0])!
	assert result.array_equal(expected)
}

fn test_pow10() {
	a := from_1d([1.0, 2, 3])!
	result := a.pow10()
	expected := from_1d([10.0, 100.0, 1000.0])!
	assert result.array_equal(expected)
}

fn test_radians() {
	a := from_1d([1.0, 2, 3])!
	result := a.radians()
	expected := from_1d([0.017453292519943295, 0.03490658503988659, 0.05235987755982989])!
	assert result.array_equal(expected)
}

fn test_round() {
	a := from_1d([1.0, 2, 3])!
	result := a.round()
	expected := from_1d([1.0, 2.0, 3.0])!
	assert result.array_equal(expected)
}

fn test_sin() {
	a := from_1d([1.0, 2, 3])!
	result := a.sin()
	expected := from_1d([0.8414709848078965, 0.9092974268256817, 0.1411200080598672])!
	assert result.array_equal(expected)
}

fn test_sinh() {
	a := from_1d([1.0, 2, 3])!
	result := a.sinh()
	expected := from_1d([1.1752011936438014, 3.626860407847019, 10.017874927409903])!
	assert result.array_equal(expected)
}

fn test_sqrt() {
	a := from_1d([1.0, 2, 3])!
	result := a.sqrt()
	expected := from_1d([1.0, 1.414213562373095, 1.7320508075688772])!
	assert result.array_equal(expected)
}

fn test_tan() {
	a := from_1d([1.0, 2, 3])!
	result := a.tan()
	expected := from_1d([1.557407724654902, -2.185039863261519, -0.1425465430742778])!
	assert result.array_equal(expected)
}

fn test_tanh() {
	a := from_1d([1.0, 2, 3])!
	result := a.tanh()
	expected := from_1d([0.7615941559557649, 0.9640275800758169, 0.9950547536867305])!
	assert result.array_equal(expected)
}

fn test_trunc() {
	a := from_1d([1.0, 2, 3])!
	result := a.trunc()
	expected := from_1d([1.0, 2.0, 3.0])!
	assert result.array_equal(expected)
}
