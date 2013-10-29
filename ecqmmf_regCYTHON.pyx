from cython.view cimport memoryview
from cython.view cimport array as cvarray
import numpy as np

cdef extern from "ecqmmf_regCPP.h":
    int updateRegistrationConstantModels(double *fixed, double *moving, double *probs, int nrows, int ncols, int nclasses, double *meansFixed, double *meansMoving, double *variancesFixed, double *variancesMoving, double *tbuffer)
    void computeMaskedWeightedImageClassStats(int *mask, double *img, double *probs, int *dims, int *labels, double *means, double *variances, double *tbuffer)
    void integrateRegistrationProbabilisticWeightedTensorFieldProductsCPP(double *q, int *dims, double *diff, int nclasses, double *probs, double *weights, double *Aw, double *bw)
    int computeRegistrationNegLogLikelihoodConstantModels(double *fixed, double *moving, int nrows, int ncols, int nclasses, 
                                    double *meansFixed, double *meansMoving, double *negLogLikelihood)
    int initializeCoupledConstantModels(double *probsFixed, double *probsMoving, int *dims, double *meansMoving)
    double optimizeECQMMFDisplacementField2DCPP(double *deltaField, double *gradientField, double *probs, int *dims, double lambda2, double *displacementField, double *residual, int maxIter, double tolerance)

cpdef update_registration_constant_models(double[:,:] fixed, double[:,:] moving, double[:,:,:,:] probs,  double[:] meansFixed, double[:] meansMoving, 
                                          double[:] variancesFixed, double[:] variancesMoving, double[:] tbuffer):
    cdef int nrows=probs.shape[0]
    cdef int ncols=probs.shape[1]
    cdef int nclasses=probs.shape[2]#==probs.shape[3]
    cdef int retVal
    retVal=updateRegistrationConstantModels(&fixed[0,0], &moving[0,0], &probs[0,0,0,0], nrows, ncols, nclasses, &meansFixed[0], &meansMoving[0], &variancesFixed[0], &variancesMoving[0], &tbuffer[0])

cpdef compute_masked_weighted_image_class_stats(int[:,:] mask, double[:,:] img, double[:,:,:] probs, int[:,:] labels):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=probs.shape[0]
    dims[1]=probs.shape[1]
    dims[2]=probs.shape[2]
    cdef double[:] means=np.zeros(shape=(dims[2],), dtype=np.double)
    cdef double[:] variances=np.zeros(shape=(dims[2],), dtype=np.double)
    cdef double[:] tbuffer=np.zeros(shape=(dims[2],), dtype=np.double)
    computeMaskedWeightedImageClassStats(&mask[0,0], &img[0,0], &probs[0,0,0], &dims[0], &labels[0,0], &means[0], &variances[0], &tbuffer[0])
    return means, variances

cpdef integrate_registration_probabilistic_weighted_tensor_field_products(double[:,:,:] q, double[:,:] diff, double[:,:,:,:] probs, double[:] weights):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=q.shape[0]
    dims[1]=q.shape[1]
    dims[2]=q.shape[2]
    cdef int k=dims[2]
    cdef int nclasses=probs.shape[2]#==probs.shape[3]
    cdef double[:,:] Aw=np.zeros(shape=(k, k), dtype=np.double)
    cdef double[:] bw=np.zeros(shape=(k,), dtype=np.double)
    integrateRegistrationProbabilisticWeightedTensorFieldProductsCPP(&q[0,0,0], &dims[0], &diff[0,0], nclasses, &probs[0,0,0,0], &weights[0], &Aw[0,0], &bw[0]);
    return Aw, bw

cpdef compute_registration_neg_log_likelihood_constant_models(double[:,:] fixed, double[:,:] moving, double[:] meansFixed, double[:] meansMoving, double[:,:,:] negLogLikelihood):
    cdef int nrows=fixed.shape[0]
    cdef int ncols=fixed.shape[1]
    cdef int nclasses=meansFixed.shape[0]
    cdef int retVal
    retVal=computeRegistrationNegLogLikelihoodConstantModels(&fixed[0,0], &moving[0,0], nrows, ncols, nclasses, &meansFixed[0], &meansMoving[0], &negLogLikelihood[0,0,0])
    
cpdef initialize_coupled_constant_models(double[:,:,:] probsFixed, double[:,:,:] probsMoving, double[:] meansMoving):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=probsFixed.shape[0]
    dims[1]=probsFixed.shape[1]
    dims[2]=probsFixed.shape[2]
    cdef int retVal
    retVal=initializeCoupledConstantModels(&probsFixed[0,0,0], &probsMoving[0,0,0], &dims[0], &meansMoving[0])

cpdef optimize_ECQMMF_displacement_field_2D(double[:,:,:] deltaField, double[:,:,:] gradientField, double[:,:,:] probs, double lambda2, double[:,:,:] displacementField, double[:,:] residual, int maxIter, double tolerance):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    dims[2]=deltaField.shape[2]
    cdef double retVal
    retVal=optimizeECQMMFDisplacementField2DCPP(&deltaField[0,0,0], &gradientField[0,0,0], &probs[0,0,0], &dims[0], lambda2, &displacementField[0,0,0], &residual[0,0], maxIter, tolerance)
    return retVal
