module num

import math

[inline]
fn num_add(a, b f64) f64 {
	return a + b
}

[inline]
fn num_subtract(a, b f64) f64 {
	return a - b
}

[inline]
fn num_divide(a, b f64) f64 {
	return a / b
}

[inline]
fn num_multiply(a, b f64) f64 {
	return a * b
}

// add adds two ndarrays elementwise
pub fn add(a, b NdArray) NdArray {
	return map2(a, b, num_add)
}

// add_aa adds two ndarrays elementwise
pub fn add_aa(a, b NdArray) NdArray {
	return add(a, b)
}

// add_as adds a scalar to an ndarray
pub fn add_as(a NdArray, b f64) NdArray {
	return map_scalar(a, b, num_add)
}

// add_sa adds an ndarray to a scalar
pub fn add_sa(a f64, b NdArray) NdArray {
	return map_scalar_lhs(a, b, num_add)
}

// subtract subtracts two ndarrays elementwise
pub fn subtract(a, b NdArray) NdArray {
	return map2(a, b, num_subtract)
}

// subtract_aa subtracts two ndarrays elementwise
pub fn subtract_aa(a, b NdArray) NdArray {
	return subtract(a, b)
}

// subtract_as subtracts a scalar from an ndarray
pub fn subtract_as(a NdArray, b f64) NdArray {
	return map_scalar(a, b, num_subtract)
}

// subtract_sa subtracts an ndarray from a scalar
pub fn subtract_sa(a f64, b NdArray) NdArray {
	return map_scalar_lhs(a, b, num_subtract)
}

// divide dives two ndarrays elementwise
pub fn divide(a, b NdArray) NdArray {
	return map2(a, b, num_divide)
}

// divide_aa divides two ndarrays elementwise
pub fn divide_aa(a, b NdArray) NdArray {
	return divide(a, b)
}

// divide_as divides an ndarray by a scalar
pub fn divide_as(a NdArray, b f64) NdArray {
	return map_scalar(a, b, num_divide)
}

// divide_sa divides a scalar by an ndarray
pub fn divide_sa(a f64, b NdArray) NdArray {
	return map_scalar_lhs(a, b, num_divide)
}

// multiplies two arrays elementwise
pub fn multiply(a, b NdArray) NdArray {
	return map2(a, b, num_multiply)
}

// multiply_aa multiples two ndarrays elementwise
pub fn multiply_aa(a, b NdArray) NdArray {
	return multiply(a, b)
}

// multiply_as multiples an ndarray by a scalar
pub fn multiply_as(a NdArray, b f64) NdArray {
	return map_scalar(a, b, num_multiply)
}

// multiply_sa multiples a scalar by an ndarray
pub fn multiply_sa(a f64, b NdArray) NdArray {
	return map_scalar_lhs(a, b, num_multiply)
}

// abs returns the absolute value elementwise of an ndarray
pub fn abs(a NdArray) NdArray {
	return amap(a, math.abs)
}

// acos returns the acos elementwise of an ndarray
pub fn acos(a NdArray) NdArray {
	return amap(a, math.acos)
}

// asin returns the elementwise asin of an ndarray
pub fn asin(a NdArray) NdArray {
	return amap(a, math.asin)
}

// atan returns the elementwise atan of an ndarray
pub fn atan(a NdArray) NdArray {
	return amap(a, math.atan)
}

// atan2 returns the atan2 elementwise of two ndarrays
pub fn atan2(a, b NdArray) NdArray {
	return map2(a, b, math.atan2)
}

// atan2_aa returns the atan2 elementwise of two ndarrays
pub fn atan2_aa(a, b NdArray) NdArray {
	return map2(a, b, math.atan2)
}

// atan2_as returns the atan2 of an ndarray and a scalar
pub fn atan2_as(a NdArray, b f64) NdArray {
	return map_scalar(a, b, math.atan2)
}

// atan2_sa returns the atan2 of a scalar and an ndarray
pub fn atan2_sa(a f64, b NdArray) NdArray {
	return map_scalar_lhs(a, b, math.atan2)
}

// cbrt returns the cube root elementwise of an ndarray
pub fn cbrt(a NdArray) NdArray {
	return amap(a, math.cbrt)
}

// ceil returns the elementwise ceil of an ndarray
pub fn ceil(a NdArray) NdArray {
	return amap(a, math.ceil)
}

// cos returns the elementwise cos of an ndarray
pub fn cos(a NdArray) NdArray {
	return amap(a, math.cos)
}

// cosh returns the elementwise cosh of an ndarray
pub fn cosh(a NdArray) NdArray {
	return amap(a, math.cosh)
}

// degrees returns elementwise rad2deg of an ndarray
pub fn degrees(a NdArray) NdArray {
	return amap(a, math.degrees)
}

// exp returns the elementwise exp of an ndarray
pub fn exp(a NdArray) NdArray {
	return amap(a, math.exp)
}

// erf returns the elementwise erf of an ndarray
pub fn erf(a NdArray) NdArray {
	return amap(a, math.erf)
}

// erfc returns the elementwise erfc of an ndarray
pub fn erfc(a NdArray) NdArray {
	return amap(a, math.erfc)
}

// exp2 returns the elementwise exp2 of an ndarray
pub fn exp2(a NdArray) NdArray {
	return amap(a, math.exp2)
}

// floor returns the elementwise floor of an ndarray
pub fn floor(a NdArray) NdArray {
	return amap(a, math.floor)
}

// fmod returns the elementwise fmod of two ndarrays
pub fn fmod(a, b NdArray) NdArray {
	return map2(a, b, math.fmod)
}

// fmod_aa returns the elementwise fmod of two ndarrays
pub fn fmod_aa(a, b NdArray) NdArray {
	return map2(a, b, math.fmod)
}

// fmod_as returns the elementwise fmod of an ndarray and a scalar
pub fn fmod_as(a NdArray, b f64) NdArray {
	return map_scalar(a, b, math.fmod)
}

// fmod_sa returns the elementwise fmod of a scalar and an ndarray
pub fn fmod_sa(a f64, b NdArray) NdArray {
	return map_scalar_lhs(a, b, math.fmod)
}

// gamma returns the elementwise gamma of an ndarray
pub fn gamma(a NdArray) NdArray {
	return amap(a, math.gamma)
}

// hypot returns the elementwise hypot of two ndarrays
pub fn hypot(a, b NdArray) NdArray {
	return map2(a, b, math.hypot)
}

// hypot_aa returns the elementwise hypot of two ndarrays
pub fn hypot_aa(a, b NdArray) NdArray {
	return map2(a, b, math.hypot)
}

// hypot_as returns the hypot of an ndarray and a scalar
pub fn hypot_as(a NdArray, b f64) NdArray {
	return map_scalar(a, b, math.hypot)
}

// hypot_sa returns the hypot of a scalar and an ndarray
pub fn hypot_sa(a f64, b NdArray) NdArray {
	return map_scalar_lhs(a, b, math.hypot)
}

// log returns the elementwise natural log of an ndarray
pub fn log(a NdArray) NdArray {
	return amap(a, math.log)
}

// log2 returns the elementwise log2 of an ndarray
pub fn log2(a NdArray) NdArray {
	return amap(a, math.log2)
}

// log10 returns the elementwise log10 of an ndarray
pub fn log10(a NdArray) NdArray {
	return amap(a, math.log10)
}

// lgamma returns the elementwise lgamma of an ndarray
pub fn lgamma(a NdArray) NdArray {
	return amap(a, math.log_gamma)
}

// log_n returns the elementwise log_n of two ndarrays
pub fn log_n(a, b NdArray) NdArray {
	return map2(a, b, math.log_n)
}

// log_n_aa returns the elementwise log_n of two ndarrays
pub fn log_n_aa(a, b NdArray) NdArray {
	return log_n(a, b)
}

// log_n_as returns the elementwise log_n of an ndarray and a scalar
pub fn log_n_as(a NdArray, b f64) NdArray {
	return map_scalar(a, b, math.log_n)
}

// log_n_sa returns the elementwise log_n of a scalar and an ndarray
pub fn log_n_sa(a f64, b NdArray) NdArray {
	return map_scalar_lhs(a, b, math.log_n)
}

// maximum returns the elementwise max of two ndarrays
pub fn maximum(a, b NdArray) NdArray {
	return map2(a, b, math.max)
}

// maximum_aa returns the elementwise max of two ndarrays
pub fn maximum_aa(a, b NdArray) NdArray {
	return map2(a, b, math.max)
}

// maximum_sa returns the elementwise max of an ndarray and a scalar
pub fn maximum_as(a NdArray, b f64) NdArray {
	return map_scalar(a, b, math.max)
}

// maximum_as returns the elementwise max of a scalar and an ndarray
pub fn maximum_sa(a f64, b NdArray) NdArray {
	return map_scalar(b, a, math.max)
}

// minimum returns the elementwise min of a two ndarrays
pub fn minimum(a, b NdArray) NdArray {
	return map2(a, b, math.min)
}

// minimum_aa returns the elementwise min of two ndarrays
pub fn minimum_aa(a, b NdArray) NdArray {
	return map2(a, b, math.min)
}

// minimum_as returns the elementwise min of an ndarray and a scalar
pub fn minimum_as(a NdArray, b f64) NdArray {
	return map_scalar(a, b, math.min)
}

// minimum_sa returns the elementwise min of a scalar and an ndarray
pub fn minimum_sa(a f64, b NdArray) NdArray {
	return map_scalar(b, a, math.min)
}

// pow returns the elementwise power of two ndarrays
pub fn pow(a, b NdArray) NdArray {
	return map2(a, b, math.pow)
}

// pow_aa returns the elementwise power of two ndarrays
pub fn pow_aa(a, b NdArray) NdArray {
	return map2(a, b, math.pow)
}

// pow_as returns the elementwise power of an ndarray and a scalar
pub fn pow_as(a NdArray, b f64) NdArray {
	return map_scalar(a, b, math.pow)
}

// pow_sa returns the elementwise power of a scalar and an ndarray
pub fn pow_sa(a f64, b NdArray) NdArray {
	return map_scalar_lhs(a, b, math.pow)
}

// radians returns the elementwise deg2rad of an ndarray
pub fn radians(a NdArray) NdArray {
	return amap(a, math.radians)
}

// round rounds elements of an ndarray elementwise
pub fn round(a NdArray) NdArray {
	return amap(a, math.round)
}

// sin returns the elementwise sin of an ndarray
pub fn sin(a NdArray) NdArray {
	return amap(a, math.sin)
}

// sinh returns the elementwise sinh of an ndarray
pub fn sinh(a NdArray) NdArray {
	return amap(a, math.sinh)
}

// sqrt returns the elementwise square root of an ndarray
pub fn sqrt(a NdArray) NdArray {
	return amap(a, math.sqrt)
}

// tan returns the elementwise tan of an ndarray
pub fn tan(a NdArray) NdArray {
	return amap(a, math.tan)
}

// tanh returns the elementwise tanh of an ndarray
pub fn tanh(a NdArray) NdArray {
	return amap(a, math.tanh)
}
