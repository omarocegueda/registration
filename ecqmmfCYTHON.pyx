from cython.view cimport memoryview
from cython.view cimport array as cvarray
import numpy as np

cdef extern from "ecqmmfCPP.h":
    int updateConstantModels(double *img, double *probs, int nrows, int ncols, int nclasses, double *means, double *variances)
    int iterateMarginals(double *likelihood, double *probs, int nrows, int ncols, int nclasses, double lambdaParam, double mu, double *N, double *D)
    int computeNegLogLikelihoodConstantModels(double *img, int nrows, int ncols, int nclasses, double *means, double *variances, double *likelihood)
    int initializeConstantModels(double *img, int nrows, int ncols, int nclasses, double *means, double *variances)
    int initializeMaximumLikelihoodProbs(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs)
    int initializeNormalizedLikelihood(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs)
    int getImageModes(double *probs, int nrows, int ncols, int nclasses, double *means, double *modes)

cpdef update_constant_models(double[:,:] img, double[:,:,:] probs, double[:] means, double[:] variances):
    cdef int nrows=probs.shape[0]
    cdef int ncols=probs.shape[1]
    cdef int nclasses=probs.shape[2]
    cdef int retVal
    retVal=updateConstantModels(&img[0,0], &probs[0,0,0], nrows, ncols, nclasses, &means[0], &variances[0])
    return retVal

cpdef iterate_marginals(double[:,:,:] likelihood, double[:,:,:] probs, double lambdaParam, double mu, double[:] N, double[:] D):
    cdef int nrows=probs.shape[0]
    cdef int ncols=probs.shape[1]
    cdef int nclasses=probs.shape[2]
    cdef int retVal
    retVal=iterateMarginals(&likelihood[0,0,0], &probs[0,0,0], nrows, ncols, nclasses, lambdaParam, mu, &N[0], &D[0])
    return retVal

cpdef compute_neg_log_likelihood_constant_models(double[:,:] img, double[:] means, double[:] variances, double[:,:,:] likelihood):
    cdef int nrows=likelihood.shape[0]
    cdef int ncols=likelihood.shape[1]
    cdef int nclasses=likelihood.shape[2]
    cdef int retVal
    retVal=computeNegLogLikelihoodConstantModels(&img[0,0], nrows, ncols, nclasses, &means[0], &variances[0], &likelihood[0,0,0])
    return retVal

cpdef initialize_constant_models(double[:,:] img, int nclasses):
    cdef int nrows=img.shape[0]
    cdef int ncols=img.shape[1]
    cdef int retVal
    cdef double[:] means=np.zeros((nclasses,))
    cdef double[:] variances=np.zeros((nclasses,))
    retVal=initializeConstantModels(&img[0,0], nrows, ncols, nclasses, &means[0], &variances[0])
    return means, variances

cpdef ecqmmf(double[:,:] img, int nclasses, double lambdaParam, double mu, int maxIter, double tolerance):
    cdef int iter_count
    cdef int nrows=img.shape[0]
    cdef int ncols=img.shape[1]
    cdef double[:] N=np.zeros((nclasses, ))
    cdef double[:] D=np.zeros((nclasses, ))
    cdef double[:] means=np.zeros((nclasses,))
    cdef double[:] variances=np.zeros((nclasses,))
    cdef double [:,:,:] probs=np.zeros((nrows, ncols, nclasses))
    cdef double [:,:,:] likelihood=np.zeros((nrows, ncols, nclasses))
    cdef double [:,:] segmented=np.zeros((nrows, ncols))
    cdef int retVal
    retVal=initializeConstantModels(&img[0,0], nrows, ncols, nclasses, &means[0], &variances[0])
    computeNegLogLikelihoodConstantModels(&img[0,0], nrows, ncols, nclasses, &means[0], &variances[0], &likelihood[0,0,0])
    #initializeMaximumLikelihoodProbs(&likelihood[0,0,0], nrows, ncols, nclasses, &probs[0,0,0])
    initializeNormalizedLikelihood(&likelihood[0,0,0], nrows, ncols, nclasses, &probs[0,0,0])
    for iter_count in range(maxIter):
        print 'Iter:',iter_count,'/',maxIter
        retVal=iterateMarginals(&likelihood[0,0,0], &probs[0,0,0], nrows, ncols, nclasses, lambdaParam, mu, &N[0], &D[0])
        retVal=updateConstantModels(&img[0,0], &probs[0,0,0], nrows, ncols, nclasses, &means[0], &variances[0])
        retVal=computeNegLogLikelihoodConstantModels(&img[0,0], nrows, ncols, nclasses, &means[0], &variances[0], &likelihood[0,0,0])
    getImageModes(&probs[0,0,0], nrows, ncols, nclasses, &means[0], &segmented[0,0])
    return segmented, means, variances, probs
    
    
    