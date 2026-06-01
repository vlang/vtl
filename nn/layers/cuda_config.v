module layers

import os

// cuda_linear_enabled is true when the user opts in to CUDA for NN layers.

// cuda_linear_enabled exposes this operation as part of the public API.

// cuda_linear_enabled exposes this operation as part of the public API.
@[inline]
pub fn cuda_linear_enabled() bool {
	return os.getenv('VTL_USE_CUDA') == '1'
}

// cuda_tests_enabled gates optional GPU tests.

// cuda_tests_enabled exposes this operation as part of the public API.

// cuda_tests_enabled exposes this operation as part of the public API.
@[inline]
pub fn cuda_tests_enabled() bool {
	return os.getenv('VTL_TEST_CUDA') == '1'
}
