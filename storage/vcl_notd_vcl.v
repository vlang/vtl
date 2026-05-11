module storage

@[params]
pub struct VclStorageParams {}

pub fn (cpu &CpuStorage[T]) vcl(params VclStorageParams) !&CpuStorage[T] {
	return error(@METHOD + ':' +
		' it is needed to compile with the flag "-d vcl" to use the VCL Backend')
}
