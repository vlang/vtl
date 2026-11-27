module quantization

import vtl
import vsl.cuda

// linear_int8_f32_cuda is unavailable unless VTL is built with `-d cuda`.
pub fn linear_int8_f32_cuda(dev &cuda.CudaDevice, input &vtl.Tensor[f32], weights &vtl.Tensor[i8], input_params QuantizationParams, weight_params PerChannelParams, bias &vtl.Tensor[f32]) !&vtl.Tensor[f32] {
	return error('linear_int8_f32_cuda: compile with -d cuda to use CUDA inference')
}
