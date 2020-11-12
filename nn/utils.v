module nn

import vtl.num
import math

fn normals(rows int, cols int) num.NdArray {
	return num.random(0, 1, [rows, cols])
}

fn sigmoid(z f64) f64 {
	return f64(1) / (f64(1) + math.exp(-z))
}

fn sigmoid_prime(z f64) f64 {
	return sigmoid(z) * (f64(1) - sigmoid(z))
}

fn htan(z f64) f64 {
	return (math.exp(f64(2) * z) - 1) / (math.exp(f64(2) * z) + 1)
}

fn htan_prime(z f64) f64 {
	return f64(1) - (math.pow((math.exp(f64(2) * z) - 1) / (math.exp(f64(2) * z) + 1), 2))
}

fn sigmoid_map(n &num.NdArray) num.NdArray {
	return num.amap(n, sigmoid)
}

fn sigmoid_prime_map(n &num.NdArray) num.NdArray {
	return num.amap(n, sigmoid_prime)
}

fn htan_map(n &num.NdArray) num.NdArray {
	return num.amap(n, htan)
}

fn htan_prime_map(n &num.NdArray) num.NdArray {
	return num.amap(n, htan_prime)
}

fn relu_map(n &num.NdArray) num.NdArray {
	return num.maximum_as(n, 0)
}

fn relu_prime(z f64) f64 {
	if z < 0 {
		return 0
	} else {
		return 1
	}
}

fn relu_prime_map(n &num.NdArray) num.NdArray {
	return num.amap(n, relu_prime)
}

fn softplus(x f64) f64 {
	return math.log(f64(1) + math.pow(math.e, x))
}

fn softplus_prime(x f64) f64 {
	return f64(1) / (f64(1) + math.pow(math.e, -x))
}

fn softplus_map(n &num.NdArray) num.NdArray {
	return num.amap(n, softplus)
}

fn softplus_prime_map(n &num.NdArray) num.NdArray {
	return num.amap(n, softplus_prime)
}
