module quantization

import vtl
import vsl.cuda

// conv2d_int8_f32_cuda is unavailable unless VTL is built with `-d cuda`.
pub fn conv2d_int8_f32_cuda(dev &cuda.CudaDevice, input &vtl.Tensor[f32], weights &vtl.Tensor[i8], input_params QuantizationParams, weight_params PerChannelParams, bias &vtl.Tensor[f32], kernel_size []int, config QuantizedConv2DConfig) !&vtl.Tensor[f32] {
	return error('conv2d_int8_f32_cuda: compile with -d cuda to use CUDA inference')
}
