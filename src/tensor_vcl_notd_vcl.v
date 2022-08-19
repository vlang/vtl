module vtl

// vcl returns a VclTensor from a Tensor
pub fn (t &Tensor<T>) vcl(params _VAnonStruct1) ?&Tensor<T> {
	return error(@STRUCT + '.' + @FN + ':' +
		' it is needed to compile with the flag "-d vcl" to use the VCL Backend')
}
