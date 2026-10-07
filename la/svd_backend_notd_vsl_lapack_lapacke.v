module la

// matrix_svd_f64 uses the pure V Jacobi implementation by default.
fn matrix_svd_f64(data []f64, rows int, columns int, full bool) !SvdFactors {
	// A wide matrix is decomposed through its transpose, keeping the Jacobi
	// iteration's working column count no larger than its row count.
	if rows < columns {
		mut transposed := []f64{len: rows * columns}
		for row in 0 .. rows {
			for column in 0 .. columns {
				transposed[column * rows + row] = data[row * columns + column]
			}
		}
		tall := matrix_svd_tall(transposed, columns, rows, full)!
		k := rows
		u_columns := if full { rows } else { k }
		vt_rows := if full { columns } else { k }
		mut u := []f64{len: rows * u_columns}
		// V of A^T is U of A.
		for row in 0 .. rows {
			for column in 0 .. u_columns {
				u[row * u_columns + column] = tall.vt[column * rows + row]
			}
		}
		mut vt := []f64{len: vt_rows * columns}
		// U of A^T transposed is V^T of A.
		for row in 0 .. vt_rows {
			for column in 0 .. columns {
				vt[row * columns + column] = tall.u[column * vt_rows + row]
			}
		}
		return SvdFactors{ values: tall.values, u: u, vt: vt }
	}
	return matrix_svd_tall(data, rows, columns, full)
}
