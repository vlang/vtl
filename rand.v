module vtl

import math
import rand
import rand.config

// RandomGenerator owns an independent pseudorandom stream for reproducible
// experiments that must not modify or depend on V's global random state.
pub struct RandomGenerator {
mut:
	rng &rand.PRNG
}

// new_random_generator creates an independent generator initialized from seed.
pub fn new_random_generator(seed int) RandomGenerator {
	mut rng := rand.new_default(config.PRNGConfigStruct{
		seed_: [u32(seed), 0]
	})
	return RandomGenerator{
		rng: rng
	}
}

// free releases the generator's underlying pseudorandom engine.
pub fn (mut generator RandomGenerator) free() {
	generator.rng.free()
}

// uniform returns f64 values in the [minimum, maximum) range.
pub fn (mut generator RandomGenerator) uniform(minimum f64, maximum f64, shape []int) !&Tensor[f64] {
	if maximum < minimum {
		return error('uniform: maximum must be greater than or equal to minimum')
	}
	mut result := zeros[f64](shape, TensorData{})
	for i in 0 .. result.size {
		result.set_nth(i, generator.rng.f64_in_range(minimum, maximum)!)
	}
	return result
}

// normal returns f64 values from a normal distribution with the given parameters.
pub fn (mut generator RandomGenerator) normal(shape []int, params NormalTensorData) !&Tensor[f64] {
	if params.sigma <= 0 {
		return error('normal: sigma must be positive')
	}
	mut result := zeros[f64](shape, TensorData{})
	for i in 0 .. result.size {
		result.set_nth(i, generator.rng.normal(config.NormalConfigStruct{
			mu:    params.mu
			sigma: params.sigma
		})!)
	}
	return result
}

// lognormal returns samples whose natural logarithm follows a normal
// distribution with the given mean and standard deviation.
pub fn (mut generator RandomGenerator) lognormal(mean f64, sigma f64, shape []int) !&Tensor[f64] {
	if math.is_nan(mean) || math.is_inf(mean, 0) || sigma < 0 || math.is_nan(sigma)
		|| math.is_inf(sigma, 0) {
		return error('lognormal: mean must be finite and sigma must be finite and non-negative')
	}
	mut result := zeros[f64](shape, TensorData{})
	for i in 0 .. result.size {
		log_value := if sigma == 0 {
			mean
		} else {
			generator.rng.normal(config.NormalConfigStruct{
				mu:    mean
				sigma: sigma
			})!
		}
		result.set_nth(i, math.exp(log_value))
	}
	return result
}

// bernoulli returns bool values sampled with the given probability of true.
pub fn (mut generator RandomGenerator) bernoulli(probability f64, shape []int) !&Tensor[bool] {
	if probability < 0 || probability > 1 {
		return error('bernoulli: probability must be in [0, 1]')
	}
	mut result := zeros[bool](shape, TensorData{})
	for i in 0 .. result.size {
		result.set_nth(i, generator.rng.bernoulli(probability)!)
	}
	return result
}

// binomial returns the number of successful trials in each sample using this
// generator's independent stream.
pub fn (mut generator RandomGenerator) binomial(trials int, probability f64, shape []int) !&Tensor[int] {
	if trials < 0 {
		return error('binomial: trials must be non-negative')
	}
	if probability < 0 || probability > 1 || math.is_nan(probability) || math.is_inf(probability, 0) {
		return error('binomial: probability must be finite and in [0, 1]')
	}
	mut result := zeros[int](shape, TensorData{})
	for i in 0 .. result.size {
		result.set_nth(i, generator.rng.binomial(trials, probability)!)
	}
	return result
}

// exponential returns f64 samples from an exponential distribution with the
// given positive finite rate using this generator's independent stream.
pub fn (mut generator RandomGenerator) exponential(lambda f64, shape []int) !&Tensor[f64] {
	if lambda <= 0 || math.is_nan(lambda) || math.is_inf(lambda, 0) {
		return error('exponential: lambda must be finite and positive')
	}
	mut result := zeros[f64](shape, TensorData{})
	for i in 0 .. result.size {
		result.set_nth(i, generator.rng.exponential(lambda))
	}
	return result
}

// geometric returns the number of Bernoulli trials needed for the first
// success, independently sampled from this generator. Results start at 1.
pub fn (mut generator RandomGenerator) geometric(probability f64, shape []int) !&Tensor[int] {
	validate_geometric_probability(probability)!
	mut result := zeros[int](shape, TensorData{})
	for i in 0 .. result.size {
		u := generator.rng.f64_in_range(0.0, 1.0)!
		result.set_nth(i, geometric_sample(probability, u))
	}
	return result
}

// choice samples values from a tensor's flattened logical order. Without
// replacement, selected positions are unique; the same value may still occur
// more than once in the population.
pub fn (mut generator RandomGenerator) choice[T](population &Tensor[T], size int, replace bool) !&Tensor[T] {
	if population.size == 0 {
		return error('choice: population must not be empty')
	}
	if size < 0 {
		return error('choice: sample size must be non-negative')
	}
	if !replace && size > population.size {
		return error('choice: cannot sample more positions than the population without replacement')
	}
	mut selected := []T{len: size}
	if replace {
		for i in 0 .. size {
			index := int(generator.rng.f64_in_range(0.0, f64(population.size))!)
			selected[i] = population.get_nth(index)
		}
	} else {
		mut positions := []int{len: population.size}
		for index in 0 .. population.size {
			positions[index] = index
		}
		for i in 0 .. size {
			offset := int(generator.rng.f64_in_range(0.0, f64(population.size - i))!)
			selected_position := i + offset
			positions[i], positions[selected_position] = positions[selected_position], positions[i]
			selected[i] = population.get_nth(positions[i])
		}
	}
	return from_1d[T](selected)
}

// permutation returns the integers in [0, size) in a seeded random order.
pub fn (mut generator RandomGenerator) permutation(size int) !&Tensor[int] {
	if size < 0 {
		return error('permutation: size must be non-negative')
	}
	mut values := []int{len: size}
	for i in 0 .. size {
		values[i] = i
	}
	for i := size - 1; i > 0; i-- {
		j := int(generator.rng.f64_in_range(0.0, f64(i + 1))!)
		values[i], values[j] = values[j], values[i]
	}
	return from_1d[int](values)
}

// gamma returns samples from a Gamma distribution using this generator's
// independent stream. `alpha` is the shape and `scale` is the scale parameter.
pub fn (mut generator RandomGenerator) gamma(alpha f64, scale f64, shape []int) !&Tensor[f64] {
	validate_gamma_parameters(alpha, scale)!
	mut rng := generator.rng
	mut values := []f64{len: size_from_shape(shape)}
	for i in 0 .. values.len {
		values[i] = sample_gamma(alpha, scale, mut rng)!
	}
	return from_array[f64](values, shape)
}

fn validate_gamma_parameters(alpha f64, scale f64) ! {
	if alpha <= 0 || scale <= 0 || math.is_nan(alpha) || math.is_inf(alpha, 0)
		|| math.is_nan(scale) || math.is_inf(scale, 0) {
		return error('gamma: alpha and scale must be finite and positive')
	}
}

fn sample_gamma(alpha f64, scale f64, mut rng &rand.PRNG) !f64 {
	shape := if alpha < 1 { alpha + 1 } else { alpha }
	d := shape - 1.0 / 3.0
	c := 1.0 / math.sqrt(9 * d)
	for _ in 0 .. 10000 {
		x := rng.normal(config.NormalConfigStruct{})!
		base := 1 + c * x
		if base <= 0 {
			continue
		}
		v := base * base * base
		u := rng.f64_in_range(0.0, 1.0)!
		if u < 1 - 0.0331 * x * x * x * x
			|| math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)) {
			mut sample := d * v
			if alpha < 1 {
				sample *= math.pow(rng.f64_in_range(0.0, 1.0)!, 1 / alpha)
			}
			return sample * scale
		}
	}
	return error('gamma: rejection sampler did not converge')
}

// beta returns samples from a Beta distribution using this generator's
// independent stream. `alpha` and `beta` are the two positive shape parameters.
pub fn (mut generator RandomGenerator) beta(alpha f64, beta f64, shape []int) !&Tensor[f64] {
	validate_gamma_parameters(alpha, 1)!
	validate_gamma_parameters(beta, 1)!
	mut rng := generator.rng
	mut values := []f64{len: size_from_shape(shape)}
	for i in 0 .. values.len {
		x := sample_gamma(alpha, 1, mut rng)!
		y := sample_gamma(beta, 1, mut rng)!
		if x + y == 0 {
			return error('beta: sampled gamma values underflowed to zero')
		}
		values[i] = x / (x + y)
	}
	return from_array[f64](values, shape)
}

// bernoulli returns a tensor of bernoulli random variables.
pub fn bernoulli[T](prob f64, shape []int, params TensorData) &Tensor[T] {
	mut t := zeros[T](shape, params)
	mut iter := t.iterator()
	for {
		_, i := iter.next() or { break }
		rand_value := cast[T](rand.bernoulli(prob) or { 0.0 })
		t.set(i, rand_value)
	}
	return t
}

// binomial returns a tensor of binomial random variables.
pub fn binomial[T](n int, prob f64, shape []int, params TensorData) !&Tensor[T] {
	mut t := zeros[T](shape, params)
	mut iter := t.iterator()
	for {
		_, i := iter.next() or { break }
		rand_value := cast[T](rand.binomial(n, prob) or { return err })
		t.set(i, rand_value)
	}
	return t
}

// exponential returns a tensor of exponential random variables.
pub fn exponential[T](lambda f64, shape []int, params TensorData) &Tensor[T] {
	mut t := zeros[T](shape, params)
	mut iter := t.iterator()
	for {
		_, i := iter.next() or { break }
		rand_value := cast[T](rand.exponential(lambda))
		t.set(i, rand_value)
	}
	return t
}

// geometric returns the number of Bernoulli trials needed for the first
// success, sampled from V's global random stream. Results start at 1.
pub fn geometric(probability f64, shape []int, params TensorData) !&Tensor[int] {
	validate_geometric_probability(probability)!
	mut result := zeros[int](shape, params)
	for i in 0 .. result.size {
		u := rand.f64_in_range(0.0, 1.0)!
		result.set_nth(i, geometric_sample(probability, u))
	}
	return result
}

fn validate_geometric_probability(probability f64) ! {
	if probability <= 0 || probability > 1 || math.is_nan(probability) || math.is_inf(probability,
		0) {
		return error('geometric: probability must be finite and in (0, 1]')
	}
}

fn geometric_sample(probability f64, uniform_value f64) int {
	if probability == 1 {
		return 1
	}
	return int(math.floor(math.log(1 - uniform_value) / math.log(1 - probability))) + 1
}

// NormalTensorData is the data for a normal distribution.

// NormalTensorData defines a public data structure for this module.

// NormalTensorData defines a public data structure for this module.
@[params]
pub struct NormalTensorData {
	TensorData
	config.NormalConfigStruct
}

// normal returns a tensor of normal random variables.
pub fn normal[T](shape []int, params NormalTensorData) &Tensor[T] {
	mut t := zeros[T](shape, memory: params.memory)
	mut iter := t.iterator()
	for {
		_, i := iter.next() or { break }
		rand_value := cast[T](rand.normal(mu: params.mu, sigma: params.sigma) or { math.nan() })
		t.set(i, rand_value)
	}
	return t
}

// random returns a new Tensor of given shape and type, initialized
// with random numbers between a given min and max value
pub fn random[T](min T, max T, shape []int, params TensorData) &Tensor[T] {
	mut t := zeros[T](shape, params)
	mut iter := t.iterator()
	for {
		_, i := iter.next() or { break }
		rand_value := random_in_range[T](min, max)
		t.set(i, rand_value)
	}
	return t
}

// random_seed resets VTL's global random generator for reproducible results.
// The seed affects random tensor constructors and model parameter
// initialization that use V's default random generator.
pub fn random_seed(i int) {
	rand.seed([u32(i), 0])
}

fn random_in_range[T](min T, max T) T {
	$if T is u16 {
		return u16(rand.int_in_range(int(min), int(max)) or { u16(min) })
	}
	$if T is u8 {
		return u8(rand.int_in_range(int(min), int(max)) or { int(min) })
	}
	$if T is u32 {
		return rand.u32_in_range(min, max) or { min }
	}
	$if T is u64 {
		return rand.u64_in_range(min, max) or { min }
	}
	$if T is i8 {
		return i8(rand.int_in_range(int(min), int(max)) or { i8(min) })
	}
	$if T is i16 {
		return i16(rand.int_in_range(int(min), int(max)) or { int(min) })
	}
	$if T is i32 {
		return i32(rand.int_in_range(int(min), int(max)) or { int(min) })
	}
	$if T is int {
		return rand.int_in_range(min, max) or { min }
	}
	$if T is i64 {
		return rand.i64_in_range(min, max) or { min }
	}
	$if T is f32 {
		return rand.f32_in_range(min, max) or { min }
	}
	$if T is f64 {
		return rand.f64_in_range(min, max) or { min }
	}
	return min
}
