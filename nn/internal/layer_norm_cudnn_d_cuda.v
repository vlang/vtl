module internal

import vsl.cuda
import vtl
import vtl.autograd_cuda
import math

// LayerNorm backward on cuDNN through the graph (backend) API (ADR-002, phase 1).
// Float32 only: cuDNN 9.27 returns no engines for FP64 LayerNorm backward, so FP64 stays on CPU.
// The legacy cudnnNormalizationBackward has no LAYER mode and cannot express the
// per-feature gamma/beta of LayerNorm, so this path builds the NORM_BACKWARD
// operation graph explicitly. The entry points are exported by libcudnn 9.x but
// their prototypes are not in the installed headers, so they are declared here.

// Enum values from cudnn_graph_v9.h (cuDNN 9.27).
const cudnn_backend_tensor_descriptor = 17
const cudnn_backend_operation_norm_backward_descriptor = 30
const cudnn_backend_operationgraph_descriptor = 15
const cudnn_backend_engineheur_descriptor = 4
const cudnn_backend_enginecfg_descriptor = 3
const cudnn_backend_execution_plan_descriptor = 5
const cudnn_backend_variant_pack_descriptor = 16

// CUDNN_DATA_FLOAT. The graph API has no FP64 LayerNorm backward engines in cuDNN 9.27.
const cudnn_data_type_float = 0

const cudnn_type_handle = 0
const cudnn_type_data_type = 1
const cudnn_type_boolean = 2
const cudnn_type_int64 = 3
const cudnn_type_void_ptr = 6
const cudnn_type_heur_mode = 8
const cudnn_type_norm_mode = 27
const cudnn_type_backend_descriptor = 15

const cudnn_attr_tensor_byte_alignment = 900
const cudnn_attr_tensor_data_type = 901
const cudnn_attr_tensor_dimensions = 902
const cudnn_attr_tensor_strides = 903
const cudnn_attr_tensor_unique_id = 906
const cudnn_attr_tensor_is_virtual = 907
const cudnn_attr_operation_norm_bwd_mode = 2100
const cudnn_attr_operation_norm_bwd_xdesc = 2101
const cudnn_attr_operation_norm_bwd_mean_desc = 2102
const cudnn_attr_operation_norm_bwd_inv_variance_desc = 2103
const cudnn_attr_operation_norm_bwd_dydesc = 2104
const cudnn_attr_operation_norm_bwd_scale_desc = 2105
const cudnn_attr_operation_norm_bwd_dscale_desc = 2107
const cudnn_attr_operation_norm_bwd_dbias_desc = 2108
const cudnn_attr_operation_norm_bwd_dxdesc = 2109
const cudnn_attr_operationgraph_handle = 800
const cudnn_attr_operationgraph_ops = 801
const cudnn_attr_engineheur_mode = 200
const cudnn_attr_engineheur_operation_graph = 201
const cudnn_attr_engineheur_results = 202
const cudnn_attr_execution_plan_handle = 400
const cudnn_attr_execution_plan_engine_config = 401
const cudnn_attr_execution_plan_workspace_size = 402
const cudnn_attr_variant_pack_unique_ids = 1000
const cudnn_attr_variant_pack_data_pointers = 1001
const cudnn_attr_variant_pack_workspace = 1003

const cudnn_norm_mode_layer = 0
const cudnn_heur_mode_instant = 0
const cudnn_heur_mode_b = 1
const cudnn_heur_mode_fallback = 2
const cudnn_heur_mode_a = 3

// Unique IDs of the eight tensors in the LayerNorm backward graph.
const ln_uid_x = i64(1)
const ln_uid_dy = i64(2)
const ln_uid_mean = i64(3)
const ln_uid_inv_variance = i64(4)
const ln_uid_scale = i64(5)
const ln_uid_dx = i64(6)
const ln_uid_dscale = i64(7)
const ln_uid_dbias = i64(8)

// Upper bound on engine configurations requested from the heuristics.
const ln_max_engine_results = 8

fn C.cudnnBackendCreateDescriptor(descriptor_type int, descriptor &voidptr) int

fn C.cudnnBackendSetAttribute(descriptor voidptr, attribute_name int, attribute_type int, element_count i64, array_of_elements voidptr) int

fn C.cudnnBackendGetAttribute(descriptor voidptr, attribute_name int, attribute_type int, requested_element_count i64, element_count &i64, array_of_elements voidptr) int

fn C.cudnnBackendFinalize(descriptor voidptr) int

fn C.cudnnBackendDestroyDescriptor(descriptor voidptr) int

fn C.cudnnBackendExecute(handle voidptr, execution_plan voidptr, variant_pack voidptr) int

// layer_norm_cudnn_eligible reports whether LayerNorm backward may run on cuDNN.
fn layer_norm_cudnn_eligible() bool {
	return autograd_cuda.cuda_backward_enabled()
}

// ln_graph is the set of backend descriptors that one LayerNorm backward call creates.
// Every descriptor is recorded so that it is destroyed on every exit path.
struct LnGraph {
mut:
	owned []voidptr
}

fn (mut g LnGraph) create(descriptor_type int) !voidptr {
	mut desc := unsafe { voidptr(nil) }
	cudnn_check(C.cudnnBackendCreateDescriptor(descriptor_type, &desc), 'cudnnBackendCreateDescriptor')!
	g.owned << desc
	return desc
}

fn (mut g LnGraph) finalize(desc voidptr, what string) ! {
	cudnn_check(C.cudnnBackendFinalize(desc), 'cudnnBackendFinalize(${what})')!
}

fn (g &LnGraph) release() {
	for desc in g.owned {
		C.cudnnBackendDestroyDescriptor(desc)
	}
}

fn ln_set_descriptor(desc voidptr, name int, value voidptr) ! {
	mut elements := [value]
	cudnn_check(C.cudnnBackendSetAttribute(desc, name, cudnn_type_backend_descriptor, 1, elements.data),
		'cudnnBackendSetAttribute(descriptor ${name})')!
}

fn ln_set_int64s(desc voidptr, name int, values []i64) ! {
	cudnn_check(C.cudnnBackendSetAttribute(desc, name, cudnn_type_int64, i64(values.len), values.data),
		'cudnnBackendSetAttribute(int64 ${name})')!
}

// ln_tensor creates a finalized float64 tensor descriptor with the given unique ID.
fn (mut g LnGraph) tensor(uid i64, dims []i64, strides []i64) !voidptr {
	desc := g.create(cudnn_backend_tensor_descriptor)!
	mut alignment := i64(16)
	cudnn_check(C.cudnnBackendSetAttribute(desc, cudnn_attr_tensor_byte_alignment, cudnn_type_int64,
		1, &alignment), 'cudnnBackendSetAttribute(alignment)')!
	mut data_type := cudnn_data_type_float
	cudnn_check(C.cudnnBackendSetAttribute(desc, cudnn_attr_tensor_data_type, cudnn_type_data_type,
		1, &data_type), 'cudnnBackendSetAttribute(data type)')!
	ln_set_int64s(desc, cudnn_attr_tensor_dimensions, dims)!
	ln_set_int64s(desc, cudnn_attr_tensor_strides, strides)!
	mut unique_id := uid
	cudnn_check(C.cudnnBackendSetAttribute(desc, cudnn_attr_tensor_unique_id, cudnn_type_int64,
		1, &unique_id), 'cudnnBackendSetAttribute(unique id)')!
	mut is_virtual := false
	cudnn_check(C.cudnnBackendSetAttribute(desc, cudnn_attr_tensor_is_virtual, cudnn_type_boolean,
		1, &is_virtual), 'cudnnBackendSetAttribute(is virtual)')!
	g.finalize(desc, 'tensor ${uid}')!
	return desc
}

// layer_norm_backward_cuda_f32 computes LayerNorm gradients on cuDNN for the
// normalized_size block layout used by layer_norm_backward_shape: the input is viewed
// as (groups, normalized_size, 1, 1), and mean and inverse standard deviation are
// computed per group on the host, as in the CPU path.
fn layer_norm_backward_cuda_f32(gradient &vtl.Tensor[f32], input &vtl.Tensor[f32], gamma &vtl.Tensor[f32], eps f64, normalized_size int) ![]&vtl.Tensor[f32] {
	if gradient.shape != input.shape {
		return error('layer_norm cuDNN backward: gradient shape must match input shape')
	}
	groups := input.size / normalized_size
	input_values := input.to_array()
	grad_values := gradient.to_array()

	mut mean_values := []f64{len: groups}
	mut inv_std_values := []f64{len: groups}
	for group in 0 .. groups {
		start := group * normalized_size
		mut mean := f64(0)
		for offset in 0 .. normalized_size {
			mean += f64(input_values[start + offset])
		}
		mean /= f64(normalized_size)
		mut variance := f64(0)
		for offset in 0 .. normalized_size {
			diff := f64(input_values[start + offset]) - mean
			variance += diff * diff
		}
		variance /= f64(normalized_size)
		mean_values[group] = mean
		inv_std_values[group] = 1.0 / math.sqrt(variance + eps)
	}

	mut dev := cuda.get_default_device()!
	defer {
		dev.release() or {}
	}
	mut graph := LnGraph{}
	defer {
		graph.release()
	}

	g := i64(groups)
	k := i64(normalized_size)
	x_dims := [g, k, 1, 1]
	x_strides := [k, 1, 1, 1]
	stat_dims := [g, i64(1), 1, 1]
	stat_strides := [i64(1), 1, 1, 1]
	affine_dims := [i64(1), k, 1, 1]
	affine_strides := [k, 1, 1, 1]

	x_desc := graph.tensor(ln_uid_x, x_dims, x_strides)!
	dy_desc := graph.tensor(ln_uid_dy, x_dims, x_strides)!
	mean_desc := graph.tensor(ln_uid_mean, stat_dims, stat_strides)!
	inv_desc := graph.tensor(ln_uid_inv_variance, stat_dims, stat_strides)!
	scale_desc := graph.tensor(ln_uid_scale, affine_dims, affine_strides)!
	dx_desc := graph.tensor(ln_uid_dx, x_dims, x_strides)!
	dscale_desc := graph.tensor(ln_uid_dscale, affine_dims, affine_strides)!
	dbias_desc := graph.tensor(ln_uid_dbias, affine_dims, affine_strides)!

	mut mode := cudnn_norm_mode_layer
	op := graph.create(cudnn_backend_operation_norm_backward_descriptor)!
	cudnn_check(C.cudnnBackendSetAttribute(op, cudnn_attr_operation_norm_bwd_mode, cudnn_type_norm_mode,
		1, &mode), 'cudnnBackendSetAttribute(norm mode)')!
	ln_set_descriptor(op, cudnn_attr_operation_norm_bwd_xdesc, x_desc)!
	ln_set_descriptor(op, cudnn_attr_operation_norm_bwd_mean_desc, mean_desc)!
	ln_set_descriptor(op, cudnn_attr_operation_norm_bwd_inv_variance_desc, inv_desc)!
	ln_set_descriptor(op, cudnn_attr_operation_norm_bwd_dydesc, dy_desc)!
	ln_set_descriptor(op, cudnn_attr_operation_norm_bwd_scale_desc, scale_desc)!
	ln_set_descriptor(op, cudnn_attr_operation_norm_bwd_dscale_desc, dscale_desc)!
	ln_set_descriptor(op, cudnn_attr_operation_norm_bwd_dbias_desc, dbias_desc)!
	ln_set_descriptor(op, cudnn_attr_operation_norm_bwd_dxdesc, dx_desc)!
	graph.finalize(op, 'norm backward')!

	op_graph := graph.create(cudnn_backend_operationgraph_descriptor)!
	mut op_array := [op]
	cudnn_check(C.cudnnBackendSetAttribute(op_graph, cudnn_attr_operationgraph_ops, cudnn_type_backend_descriptor,
		1, op_array.data), 'cudnnBackendSetAttribute(operation graph ops)')!
	cudnn_check(C.cudnnBackendSetAttribute(op_graph, cudnn_attr_operationgraph_handle, cudnn_type_handle,
		1, &dev.cudnn), 'cudnnBackendSetAttribute(operation graph handle)')!
	graph.finalize(op_graph, 'operation graph')!

	mut configs := []voidptr{len: ln_max_engine_results, init: unsafe { nil }}
	for i in 0 .. configs.len {
		configs[i] = graph.create(cudnn_backend_enginecfg_descriptor)!
	}
	// The heuristic mode limits which engines may be returned. INSTANT is tried first;
	// the broader modes are used when it finds no engine for this graph.
	mut result_count := i64(0)
	for heur_mode in [cudnn_heur_mode_instant, cudnn_heur_mode_a, cudnn_heur_mode_b,
		cudnn_heur_mode_fallback] {
		heur := graph.create(cudnn_backend_engineheur_descriptor)!
		mut heur_mode_value := heur_mode
		cudnn_check(C.cudnnBackendSetAttribute(heur, cudnn_attr_engineheur_mode, cudnn_type_heur_mode,
			1, &heur_mode_value), 'cudnnBackendSetAttribute(heur mode)')!
		ln_set_descriptor(heur, cudnn_attr_engineheur_operation_graph, op_graph)!
		graph.finalize(heur, 'engine heuristics')!
		cudnn_check(C.cudnnBackendGetAttribute(heur, cudnn_attr_engineheur_results, cudnn_type_backend_descriptor,
			i64(configs.len), &result_count, configs.data), 'cudnnBackendGetAttribute(heur results)')!
		if result_count > 0 {
			break
		}
	}
	if result_count == 0 {
		return error('layer_norm cuDNN backward: no engine configuration for this shape')
	}

	// The heuristics rank configurations best first. Use the first one that finalizes
	// into an execution plan.
	mut plan := unsafe { voidptr(nil) }
	mut plan_error := ''
	for i in 0 .. int(result_count) {
		candidate := graph.create(cudnn_backend_execution_plan_descriptor)!
		cudnn_check(C.cudnnBackendSetAttribute(candidate, cudnn_attr_execution_plan_handle, cudnn_type_handle,
			1, &dev.cudnn), 'cudnnBackendSetAttribute(plan handle)')!
		ln_set_descriptor(candidate, cudnn_attr_execution_plan_engine_config, configs[i]) or {
			plan_error = err.msg()
			continue
		}
		status := C.cudnnBackendFinalize(candidate)
		if status != cuda.cudnn_status_success {
			plan_error = cuda.cudnn_error(status)
			continue
		}
		plan = candidate
		break
	}
	if plan == unsafe { nil } {
		return error('layer_norm cuDNN backward: no execution plan could be finalized (${plan_error})')
	}

	mut workspace_bytes := i64(0)
	mut workspace_count := i64(0)
	cudnn_check(C.cudnnBackendGetAttribute(plan, cudnn_attr_execution_plan_workspace_size,
		cudnn_type_int64, 1, &workspace_count, &workspace_bytes), 'cudnnBackendGetAttribute(workspace)')!

	count := input.size
	d_x := device_alloc_bytes(count * int(sizeof(f32)))!
	defer {
		C.cudaFree(d_x)
	}
	d_dy := device_alloc_bytes(count * int(sizeof(f32)))!
	defer {
		C.cudaFree(d_dy)
	}
	d_mean := device_alloc_bytes(groups * int(sizeof(f32)))!
	defer {
		C.cudaFree(d_mean)
	}
	d_inv := device_alloc_bytes(groups * int(sizeof(f32)))!
	defer {
		C.cudaFree(d_inv)
	}
	d_scale := device_alloc_bytes(normalized_size * int(sizeof(f32)))!
	defer {
		C.cudaFree(d_scale)
	}
	d_dx := device_alloc_bytes(count * int(sizeof(f32)))!
	defer {
		C.cudaFree(d_dx)
	}
	d_dscale := device_alloc_bytes(normalized_size * int(sizeof(f32)))!
	defer {
		C.cudaFree(d_dscale)
	}
	d_dbias := device_alloc_bytes(normalized_size * int(sizeof(f32)))!
	defer {
		C.cudaFree(d_dbias)
	}
	mut d_workspace := unsafe { voidptr(nil) }
	if workspace_bytes > 0 {
		d_workspace = device_alloc_bytes(int(workspace_bytes))!
	}
	defer {
		if d_workspace != unsafe { nil } {
			C.cudaFree(d_workspace)
		}
	}

	device_upload_f32(d_x, input_values)!
	device_upload_f32(d_dy, grad_values)!
	device_upload_f32(d_mean, mean_values.map(f32(it)))!
	device_upload_f32(d_inv, inv_std_values.map(f32(it)))!
	device_upload_f32(d_scale, gamma.to_array())!

	mut uids := [ln_uid_x, ln_uid_dy, ln_uid_mean, ln_uid_inv_variance, ln_uid_scale, ln_uid_dx,
		ln_uid_dscale, ln_uid_dbias]
	mut pointers := [d_x, d_dy, d_mean, d_inv, d_scale, d_dx, d_dscale, d_dbias]
	pack := graph.create(cudnn_backend_variant_pack_descriptor)!
	cudnn_check(C.cudnnBackendSetAttribute(pack, cudnn_attr_variant_pack_unique_ids, cudnn_type_int64,
		i64(uids.len), uids.data), 'cudnnBackendSetAttribute(variant unique ids)')!
	cudnn_check(C.cudnnBackendSetAttribute(pack, cudnn_attr_variant_pack_data_pointers, cudnn_type_void_ptr,
		i64(pointers.len), pointers.data), 'cudnnBackendSetAttribute(variant data pointers)')!
	mut workspace_ptr := d_workspace
	cudnn_check(C.cudnnBackendSetAttribute(pack, cudnn_attr_variant_pack_workspace, cudnn_type_void_ptr,
		1, &workspace_ptr), 'cudnnBackendSetAttribute(variant workspace)')!
	graph.finalize(pack, 'variant pack')!

	cudnn_check(C.cudnnBackendExecute(dev.cudnn, plan, pack), 'cudnnBackendExecute(layer norm backward)')!

	mut dx_values := []f32{len: count}
	device_download_f32(d_dx, mut dx_values)!
	mut dscale_values := []f32{len: normalized_size}
	device_download_f32(d_dscale, mut dscale_values)!
	mut dbias_values := []f32{len: normalized_size}
	device_download_f32(d_dbias, mut dbias_values)!

	return [
		vtl.from_array(dx_values, input.shape)!,
		vtl.from_array(dscale_values, gamma.shape)!,
		vtl.from_array(dbias_values, gamma.shape)!,
	]
}

// device_upload_f32 copies host float32 values to device memory at dst.
fn device_upload_f32(dst voidptr, src []f32) ! {
	if src.len == 0 {
		return
	}
	status := C.cudaMemcpy(dst, src.data, int(sizeof(f32)) * src.len, cuda.cuda_memcpy_host_to_device)
	if status != 0 {
		return error('cudaMemcpy host-to-device failed with status ${status}')
	}
}

// device_download_f32 copies device float32 values from src into dst.
fn device_download_f32(src voidptr, mut dst []f32) ! {
	if dst.len == 0 {
		return
	}
	status := C.cudaMemcpy(dst.data, src, int(sizeof(f32)) * dst.len, cuda.cuda_memcpy_device_to_host)
	if status != 0 {
		return error('cudaMemcpy device-to-host failed with status ${status}')
	}
}
