cimport cython
from libc.math cimport sin

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef uR3(double[:] xn, double[:] yn, double[:] k, int max,
          double[:] xnk0, double[:] ynk0, int maxk0,
          int threshold, double b,
          double R, double phi, double z, double[:] out):
    '''
    solve uR3 due to a stokeslet in a pipe.
    '''
    out[1] = 1.


cpdef try1(double a, double[:] out):
    '''
    try
    '''
    cdef double out1=0
    out1 = a
    out[0] = a
    return out1

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef clip(double[:] a, double min, double max, double[:] out):
#     '''
#     Clip the values in a to be between min and max. Result in out
#     '''
#     if min > max:
#         raise ValueError("min must be <= max")
#     if a.shape[0] != out.shape[0]:
#         raise ValueError("input and output arrays must be the same size")
#     for i in range(a.shape[0]):
#         if a[i] < min:
#             out[i] = min
#         elif a[i] > max:
#             out[i] = max
#         else:
#             out[i] = a[i]
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef clip_fast(double[:] a, double min, double max, double[:] out):
#     if min > max:
#         raise ValueError("min must be <= max")
#     if a.shape[0] != out.shape[0]:
#         raise ValueError("input and output arrays must be the same size")
#     for i in range(a.shape[0]):
#         out[i] = (a[i] if a[i] < max else max) if a[i] > min else min
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef clip2d(double[:,:] a, double min, double max, double[:,:] out):
#     if min > max:
#         raise ValueError("min must be <= max")
#     for n in range(a.ndim):
#         if a.shape[n] != out.shape[n]:
#             raise TypeError("a and out have different shapes")
#     for i in range(a.shape[0]):
#         for j in range(a.shape[1]):
#             if a[i,j] < min:
#                 out[i,j] = min
#             elif a[i,j] > max:
#                 out[i,j] = max
#             else:
#                 out[i,j] = a[i,j]
