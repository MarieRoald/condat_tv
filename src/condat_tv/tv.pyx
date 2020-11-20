import numpy


import numpy as np
cimport numpy as np


cdef extern from "../c/Condat_TV_1D_v2.c":
    void TV1D_denoise_v2(double* input, double* output, unsigned int width, const double lambda_)


# Wrapper for the C++ implementation of the Perona-Malik denoising algorithm
cdef c_tv_denoise(np.ndarray[double, ndim=1] signal,
                  np.ndarray[double, ndim=1] output,
                  unsigned int width,
                  const double regularisation_strength):
    TV1D_denoise_v2(<double *> signal.data, <double *> output.data, width, regularisation_strength)


# Python function for the Perona-Malik denoising algorithm
cpdef tv_denoise(signal, regularisation_strength, out=None):
    """
    Arguments:
    ----------
    signal: np.ndarray[double, ndim=1]
    regularisation_strength: float
    out: np.ndarray[double, ndim=1] (same size as signal, optional)
    """
    signal = np.ascontiguousarray(signal, dtype=np.double)
    if out is None:
        out = np.empty_like(signal)
    elif not out.data.c_contiguous:
        raise ValueError("Output vector must be C-contiguous")
    elif out.dtype != np.double:
        raise ValueError("Output vector must have dtype=np.double")

    c_tv_denoise(signal, out, len(signal), regularisation_strength)
    return out


cpdef tv_denoise_matrix(signal, regularisation_strength, out=None):
    """
    Apply the TV denoising operator on rows of the signal matrix

    Arguments:
    ----------
    signal : np.ndarray[double, ndim=2]
    regularisation_strength : float
    out : np.ndarray[double, ndim=2],  (same size as signal, optional)
    """
    # We need 1D arrays
    # Multidimensional numpy arrays are by default C-contiguous
    # signal[0, :] is one contiguous array
    # signal[i, :] is anoter contiguous array
    # signal[:, 0] is not a contigous array
    # signal[:, i] is also not a contiguous array 
    signal = np.ascontiguousarray(signal, dtype=np.double)

    if out is None:
        out = np.empty_like(signal)
    elif not out.data.c_contiguous:
        raise ValueError("Output matrix must be C-contiguous")
    elif out.dtype != np.double:
        raise ValueError("Output matrix must have dtype=np.double")
    
    num_rows, row_length = signal.shape
    for i in range(num_rows):
        c_tv_denoise(signal[i], out[i], row_length, regularisation_strength)

    return out
