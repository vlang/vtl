module datasets

import vtl

// TensorTransform maps one tensor to another and may report a transformation
// error. It can be applied to an individual sample or a complete batch.
pub type TensorTransform[T] = fn (input &vtl.Tensor[T]) !&vtl.Tensor[T]

// RandomTensorTransform applies a tensor transform using an injected generator.
pub type RandomTensorTransform[T] = fn (input &vtl.Tensor[T], mut generator vtl.RandomGenerator) !&vtl.Tensor[T]

// TransformPipeline applies tensor transforms in their configured order.
pub struct TransformPipeline[T] {
	transforms []TensorTransform[T]
}

// RandomTransformPipeline composes transforms that consume an explicit RNG.
pub struct RandomTransformPipeline[T] {
	transforms []RandomTensorTransform[T]
}

// TransformPipeline.new creates a pipeline and snapshots the transform list.
pub fn TransformPipeline.new[T](transforms []TensorTransform[T]) TransformPipeline[T] {
	return TransformPipeline[T]{
		transforms: transforms.clone()
	}
}

// RandomTransformPipeline.new snapshots the configured random transforms.
pub fn RandomTransformPipeline.new[T](transforms []RandomTensorTransform[T]) RandomTransformPipeline[T] {
	return RandomTransformPipeline[T]{
		transforms: transforms.clone()
	}
}

// apply passes the output of each transform to the next one. An empty pipeline
// returns the input unchanged; a transform error stops the pipeline.
pub fn (pipeline &TransformPipeline[T]) apply(input &vtl.Tensor[T]) !&vtl.Tensor[T] {
	mut output := input
	for transform in pipeline.transforms {
		output = transform(output)!
	}
	return output
}

// apply passes an explicit generator through each transform for reproducible replay.
pub fn (pipeline &RandomTransformPipeline[T]) apply(input &vtl.Tensor[T], mut generator vtl.RandomGenerator) !&vtl.Tensor[T] {
	mut output := input
	for transform in pipeline.transforms {
		output = transform(output, mut generator)!
	}
	return output
}
