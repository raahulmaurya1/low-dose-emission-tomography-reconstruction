# projector.py: operator A / A^T and the call-counting wrapper.
# Takes: image size n, angles in degrees, detector bins n_det (n + 2 by default).
# Returns: sparse A of shape (n_angles*n_det, n*n); CountingOperator with forward/back and call counts.
