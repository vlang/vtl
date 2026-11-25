module datasets

import vtl

// TensorTransform maps one tensor to another and may report a transformation
// error. It can be applied to an individual sample or a complete batch.
pub type TensorTransform[T] = fn (input &vtl.Tensor[T]) !&vtl.Tensor[T]

// TransformPipeline applies tensor transforms in their configured order.
pub struct TransformPipeline[T] {
	transforms []TensorTransform[T]
}

// TransformPipeline.new creates a pipeline and snapshots the transform list.
pub fn TransformPipeline.new[T](transforms []TensorTransform[T]) TransformPipeline[T] {
	return TransformPipeline[T]{
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
