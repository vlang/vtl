module vtl

// vcl returns a VclTensor from a Tensor
pub fn (t &Tensor<T>) vcl(params struct {}) ?&Tensor<T> {
	return error(@STRUCT + '.' + @METHOD + ':' +
		' it is needed to compile with the flag "-d vcl" to use the VCL Backend')
}
