module vtl

// Backend selects compute backend for runtime dispatch.
pub enum Backend {
	auto
	vulkan
	vcl
	cpu
}
