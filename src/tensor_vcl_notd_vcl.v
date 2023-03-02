module vtl

[params]
pub struct VclParams {}

// vcl returns a VclTensor from a Tensor
pub fn (t &Tensor[T]) vcl(params VclParams) !&Tensor[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vcl" to use the VCL Backend')
}
