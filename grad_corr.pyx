import numpy as np
cimport cython
from dipy.align.fused_types cimport floating

cdef extern from "math.h":
    double sqrt(double x) nogil


cdef inline int _int_max(int a, int b) nogil:
    r"""
    Returns the maximum of a and b
    """
    return a if a >= b else b


cdef inline int _int_min(int a, int b) nogil:
    r"""
    Returns the minimum of a and b
    """
    return a if a <= b else b


cdef enum:
    DI_DRR = 0
    DI_DRC = 1
    DI_DCC = 2
    DJ_DRR = 3
    DJ_DRC = 4
    DJ_DCC = 5


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_gc_factors_2d(floating[:, :] static, floating[:, :] moving):
    r"""
    Precomputes the entries of the Hessian matrix of static and moving at each 
    pixel of their common domain

    Parameters
    ----------
    static : array, shape (R, C)
        the current static image
    moving : array, shape (R, C)
        the current moving image

    Returns
    -------
    factors : array, shape (R, C, 6)
        the precomputed Hessian entries: 
        factors[:,:,:,0] : second derivative of static w.r.t. rows twice
        factors[:,:,:,1] : second derivative of static w.r.t. rows and colums
        factors[:,:,:,2] : second derivative of static w.r.t. colums twice
        factors[:,:,:,3] : second derivative of moving w.r.t. rows twice
        factors[:,:,:,4] : second derivative of moving w.r.t. rows and columns
        factors[:,:,:,5] : second derivative of moving w.r.t. columns tice      
    """
    cdef int nr = static.shape[0]
    cdef int nc = static.shape[1]
    cdef int i,j
    cdef double dIdrr, dIcc, dIrc, dJdrr, dJdcc, dJdrc
    cdef floating[:, :, :] factors = np.zeros((nr, nc, 6), dtype=np.asarray(static).dtype)
    for i in range(nr):
        for j in range(nc):
            if 0<i<nr-1:
                dIdrr = static[i-1, j] - 2*static[i, j] + static[i+1, j]
                dJdrr = moving[i-1, j] - 2*moving[i, j] + moving[i+1, j]
            else:
                dIdrr = 0
                dJdrr = 0

            if 0<j<nc-1:
                dIdcc = static[i, j-1] - 2*static[i, j] + static[i, j+1]
                dJdcc = moving[i, j-1] - 2*moving[i, j] + moving[i, j+1]
            else:
                dIdcc = 0
                dJdcc = 0

            if 0<i<nr-1 and 0<j<nc-1:
                dIdrc = static[i+1, j+1] + static[i, j] - static[i+1, j] - static[i, j+1]
                dJdrc = moving[i+1, j+1] + moving[i, j] - moving[i+1, j] - moving[i, j+1]
            else:
                dIdrc = 0
                dJdrc = 0
            factors[i, j, DI_DRR] = dIdrr
            factors[i, j, DI_DRC] = dIdrc
            factors[i, j, DI_DCC] = dIdcc
            factors[i, j, DJ_DRR] = dJdrr
            factors[i, j, DJ_DRC] = dJdrc
            factors[i, j, DJ_DCC] = dJdcc
    return factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_gc_forward_step_2d(floating[:, :, :] grad_static,
                               floating[:, :, :] grad_moving,
                               floating[:, :, :] factors):
    r"""
    Computes the gradient of the Gradient Correlation metric for symmetric
    registration (SyN) w.r.t. the displacement associated to the moving
    volume ('forward' step)

    Parameters
    ----------
    grad_static : array, shape (R, C, 2)
        the gradient of the static volume
    grad_moving : array, shape (R, C, 2)
        the gradient of the moving volume
    factors : array, shape (R, C, 6)
        the precomputed Hessian matrix entries

    Returns
    -------
    out : array, shape (R, C, 2)
        the gradient of the gradient correlation metric with respect to the 
        displacement associated to the moving volume
    energy : the gradient correlation energy (data term) at this iteration
    """
    cdef int nr = grad_static.shape[0]
    cdef int nc = grad_static.shape[1]
    cdef double energy = 0
    cdef double dJdrr, dJdcc, dJdrc, local_correlation, prod, temp0, temp1, ngs2, ngm2
    cdef floating[:, :, :] out = np.zeros((nr, nc, 2), 
                                             dtype=np.asarray(grad_static).dtype)

    with nogil:

        for r in range(nr):
            for c in range(nc):
                dJdrr = factors[r, c, DJ_DRR]
                dJdcc = factors[r, c, DJ_DCC]
                dJdrc = factors[r, c, DJ_DRC]

                ngs2 = grad_static[r, c, 0]**2 + grad_static[r, c, 1]**2
                ngm2 = grad_moving[r, c, 0]**2 + grad_moving[r, c, 1]**2
                
                if(ngs2 == 0.0 or ngm2 == 0.0):
                    continue
                
                prod = grad_static[r, c, 0]*grad_moving[r, c, 0] + grad_static[r, c, 1]*grad_moving[r, c, 1]
                local_correlation = 0
                
                if(ngs2 * ngm2 > 1e-5):
                    local_correlation = prod * prod / (ngs2 * ngm2)
                
                if(local_correlation < 1+1e-6):  # avoid bad values...
                    energy -= local_correlation
                
                temp0 = (2.0 * prod / (ngs2 * ngm2)) * (grad_static[r, c, 0] - (grad_moving[r, c, 0] * prod)/ngm2)
                temp1 = (2.0 * prod / (ngs2 * ngm2)) * (grad_static[r, c, 1] - (grad_moving[r, c, 1] * prod)/ngm2)
                out[r, c, 0] = dJdrr * temp0 + dJdrc * temp1
                out[r, c, 1] = dJdrc * temp0 + dJdcc * temp1
    return out, energy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_gc_backward_step_2d(floating[:, :, :] grad_static,
                                floating[:, :, :] grad_moving,
                                floating[:, :, :] factors):
    r"""
    Computes the gradient of the Gradient Correlation metric for symmetric
    registration (SyN) w.r.t. the displacement associated to the static
    volume ('backward' step)

    Parameters
    ----------
    grad_static : array, shape (R, C, 2)
        the gradient of the static volume
    grad_moving : array, shape (R, C, 2)
        the gradient of the moving volume
    factors : array, shape (R, C, 6)
        the precomputed Hessian matrix entries

    Returns
    -------
    out : array, shape (R, C, 2)
        the gradient of the gradient correlation metric with respect to the 
        displacement associated to the static volume
    energy : the gradient correlation energy (data term) at this iteration
    """
    cdef int nr = grad_static.shape[0]
    cdef int nc = grad_static.shape[1]
    cdef double energy = 0
    cdef double dIdrr, dIdcc, dIdrc, local_correlation, prod, temp0, temp1, ngs2, ngm2
    cdef floating[:, :, :] out = np.zeros((nr, nc, 2), 
                                             dtype=np.asarray(grad_static).dtype)

    with nogil:

        for r in range(nr):
            for c in range(nc):
                dIdrr = factors[r, c, DI_DRR]
                dIdcc = factors[r, c, DI_DCC]
                dIdrc = factors[r, c, DI_DRC]

                ngs2 = grad_static[r, c, 0]**2 + grad_static[r, c, 1]**2
                ngm2 = grad_moving[r, c, 0]**2 + grad_moving[r, c, 1]**2
                
                if(ngs2 == 0.0 or ngm2 == 0.0):
                    continue
                
                prod = grad_static[r, c, 0]*grad_moving[r, c, 0] + grad_static[r, c, 1]*grad_moving[r, c, 1]
                local_correlation = 0
                
                if(ngs2 * ngm2 > 1e-5):
                    local_correlation = prod * prod / (ngs2 * ngm2)
                
                if(local_correlation < 1+1e-6):  # avoid bad values...
                    energy -= local_correlation
                
                temp0 = (2.0 * prod / (ngs2 * ngm2)) * (grad_moving[r, c, 0] - (grad_static[r, c, 0] * prod)/ngs2)
                temp1 = (2.0 * prod / (ngs2 * ngm2)) * (grad_moving[r, c, 1] - (grad_static[r, c, 1] * prod)/ngs2)
                out[r, c, 0] = dIdrr * temp0 + dIdrc * temp1
                out[r, c, 1] = dIdrc * temp0 + dIdcc * temp1 
    return out, energy
