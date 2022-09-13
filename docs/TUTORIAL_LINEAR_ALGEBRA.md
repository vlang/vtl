# Tutorial: Matrix and Vector operations

The following linear algebra operations are supported for
tensors of rank 1 (vectors) and 2 (matrices):

- dot product (Vector to Vector) using `vtl.la.dot`
- addition and subtraction (any rank) using `vtl.add` and `vtl.subtract`
- multiplication or division by a scalar using `vtl.multiply` and `vtl.divide`
- matrix-matrix multiplication using `vtl.la.matmul`
- . . .

*Note*: Matrix operations for floats are accelerated using
[vsl.blas](https://github.com/vlang/vsl/tree/master/blas).
Unfortunately there is no acceleration routine for integers.
Integer matrix-matrix and matrix-vector multiplications
are implemented via semi-optimized routines.
