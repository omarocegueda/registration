from cython.view cimport memoryview
from cython.view cimport array as cvarray
import numpy as np
import ecqmmf

cdef extern from "ecqmmfCPP.h":
    int initializeMaximumLikelihoodProbs(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs)
    int initializeNormalizedLikelihood(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs)
    int iterateMarginals(double *likelihood, double *probs, int nrows, int ncols, int nclasses, double lambdaParam, double mu, double *N, double *D)

cdef extern from "ecqmmf_regCPP.h":
    int updateRegistrationConstantModels(double *fixed, double *moving, double *probs, int nrows, int ncols, int nclasses, double *meansFixed, double *meansMoving, double *variancesFixed, double *variancesMoving, double *tbuffer)
    void integrateRegistrationProbabilisticWeightedTensorFieldProductsCPP(double *q, int *dims, double *diff, int nclasses, double *probs, double *weights, double *Aw, double *bw)
    int computeRegistrationNegLogLikelihoodConstantModels(double *fixed, double *moving, double *probs, int nrows, int ncols, int nclasses, double *meansFixed, double *meansMoving, double *variancesFixed, double *variancesMoving, double *negLogLikelihood)

cpdef update_registration_constant_models(double[:,:] fixed, double[:,:] moving, double[:,:,:,:] probs,  double[:] meansFixed, double[:] meansMoving, 
                                          double[:] variancesFixed, double[:] variancesMoving, double[:] tbuffer):
    cdef int nrows=probs.shape[0]
    cdef int ncols=probs.shape[1]
    cdef int nclasses=probs.shape[2]#==probs.shape[3]
    cdef int retVal
    retVal=updateRegistrationConstantModels(&fixed[0,0], &moving[0,0], &probs[0,0,0,0], nrows, ncols, nclasses, &meansFixed[0], &meansMoving[0], &variancesFixed[0], &variancesMoving[0], &tbuffer[0])

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

cpdef compute_registration_neg_log_likelihood_constant_models(double[:,:] fixed, double[:,:] moving, double[:,:,:,:] probs, double[:] meansFixed, double[:] meansMoving, double[:] variancesFixed, double[:] variancesMoving, double[:,:,:,:] negLogLikelihood):
    cdef int nrows=probs.shape[0]
    cdef int ncols=probs.shape[1]
    cdef int nclasses=probs.shape[2]#==probs.shape[3]
    cdef int retVal
    retVal=computeRegistrationNegLogLikelihoodConstantModels(&fixed[0,0], &moving[0,0], &probs[0,0,0,0], nrows, ncols, nclasses, &meansFixed[0], &meansMoving[0], &variancesFixed[0], &variancesMoving[0], &negLogLikelihood[0,0,0,0])

cpdef initialize_registration_maximum_likelihood_probs(double[:,:,:,:] negLogLikelihood, double[:,:,:,:] probs):
    cdef int nrows=negLogLikelihood.shape[0]
    cdef int ncols=negLogLikelihood.shape[1]
    cdef int nclasses=negLogLikelihood.shape[2]
    cdef int retVal
    retVal=initializeMaximumLikelihoodProbs(&negLogLikelihood[0,0,0,0], nrows, ncols, nclasses*nclasses, &probs[0,0,0,0])

cpdef initialize_registration_normalized_likelihood(double[:,:,:,:] negLogLikelihood, double[:,:,:,:] probs):
    cdef int nrows=negLogLikelihood.shape[0]
    cdef int ncols=negLogLikelihood.shape[1]
    cdef int nclasses=negLogLikelihood.shape[2]
    cdef int retVal
    retVal=initializeNormalizedLikelihood(&negLogLikelihood[0,0,0,0], nrows, ncols, nclasses*nclasses, &probs[0,0,0,0])

cpdef iterate_marginals(double[:,:,:,:] negLogLikelihood, double[:,:,:,:] probs, lambdaParam, mu, double[:] N, double[:] D):
    cdef int nrows=negLogLikelihood.shape[0]
    cdef int ncols=negLogLikelihood.shape[1]
    cdef int nclasses=negLogLikelihood.shape[2]
    cdef int retVal
    retVal=iterateMarginals(&negLogLikelihood[0,0,0,0], &probs[0,0,0,0], nrows, ncols, nclasses*nclasses, lambdaParam, mu, &N[0], &D[0])
