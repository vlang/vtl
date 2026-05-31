module layers

import os

// cuda_linear_enabled is true when the user opts in to CUDA for NN layers.
// Requires building with `-d cuda` and setting VTL_USE_CUDA=1.
// Default is off so dev workflows stay CPU-only and avoid GPU driver load.
@[inline]
pub fn cuda_linear_enabled() bool {
	return os.getenv('VTL_USE_CUDA') == '1'
}

// cuda_tests_enabled gates optional GPU tests (see linear_cuda_test.v).
@[inline]
pub fn cuda_tests_enabled() bool {
	return os.getenv('VTL_TEST_CUDA') == '1'
}
