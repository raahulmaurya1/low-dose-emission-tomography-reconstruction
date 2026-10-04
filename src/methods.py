# methods.py: reconstruction methods fbp, mlem, osem, mlem_tv.
# Takes: a CountingOperator, sinogram m [n_angles, n_det], and method parameters.
# Returns: fbp -> image [n,n] (data units); iterative methods -> (x [n,n], history).
