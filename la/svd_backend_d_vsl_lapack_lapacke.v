module la

import vsl.lapack as vsl_lapack

// matrix_svd_f64 dispatches SVD to VSL's native LAPACKE backend when the
// optional `vsl_lapack_lapacke` build flag is enabled.
fn matrix_svd_f64(data []f64, rows int, columns int, full bool) !SvdFactors {
	k := if rows < columns { rows } else { columns }
	u_columns := if full { rows } else { k }
	vt_rows := if full { columns } else { k }
	mut input := data.clone()
	mut values := []f64{len: k}
	mut u := []f64{len: rows * u_columns}
	mut vt := []f64{len: vt_rows * columns}
	mut superb := []f64{len: if k > 0 { k - 1 } else { 0 }}
	job := if full { vsl_lapack.SVDJob.svd_all } else { vsl_lapack.SVDJob.svd_store }
	info := vsl_lapack.dgesvd(job, job, rows, columns, mut input, columns, values, mut u,
		u_columns, mut vt, columns, superb)
	if info < 0 {
		return error('LAPACKE dgesvd rejected argument ${-info}')
	}
	if info > 0 {
		return error('LAPACKE dgesvd failed to converge (info=${info})')
	}
	return SvdFactors{ values: values, u: u, vt: vt }
}
