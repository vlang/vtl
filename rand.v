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
	$if T is u32 {
		return rand.u32_in_range(min, max) or { min }
	}
	$if T is u64 {
		return rand.u64_in_range(min, max) or { min }
	}
	$if T is i8 {
		return i8(rand.int_in_range(int(min), int(max)) or { i8(min) })
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
