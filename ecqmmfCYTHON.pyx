from cython.view cimport memoryview
from cython.view cimport array as cvarray
import numpy as np

cdef extern from "ecqmmfCPP.h":
    double updateConstantModels(double *img, double *probs, int nrows, int ncols, int nclasses, double *means, double *variances)
    int updateVariances(double *img, double *probs, int nrows, int ncols, int nclasses, double *means, double *variances)
    double iterateMarginals(double *likelihood, double *probs, int nrows, int ncols, int nclasses, double lambdaParam, double mu, double *N, double *D, double *prev)
    int computeNegLogLikelihoodConstantModels(double *img, int nrows, int ncols, int nclasses, double *means, double *variances, double *likelihood)
    int initializeConstantModels(double *img, int nrows, int ncols, int nclasses, double *means, double *variances)
    int initializeMaximumLikelihoodProbs(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs)
    int initializeNormalizedLikelihood(double *negLogLikelihood, int nrows, int ncols, int nclasses, double *probs)
    int getImageModes(double *probs, int nrows, int ncols, int nclasses, double *means, double *modes)
    double optimizeMarginals(double *likelihood, double *probs, int nrows, int ncols, int nclasses, double lambdaParam, double mu, int maxIter, double tolerance)

cpdef update_constant_models(double[:,:] img, double[:,:,:] probs, double[:] means, double[:] variances):
    cdef int nrows=probs.shape[0]
    cdef int ncols=probs.shape[1]
    cdef int nclasses=probs.shape[2]
    cdef double retVal
    retVal=updateConstantModels(&img[0,0], &probs[0,0,0], nrows, ncols, nclasses, &means[0], &variances[0])
    return retVal
    
cpdef update_variances(double[:,:] img, double[:,:,:] probs, double[:] means, double[:] variances):
    cdef int nrows=probs.shape[0]
    cdef int ncols=probs.shape[1]
    cdef int nclasses=probs.shape[2]
    cdef int retVal
    retVal=updateVariances(&img[0,0], &probs[0,0,0], nrows, ncols, nclasses, &means[0], &variances[0])
    return retVal


cpdef iterate_marginals(double[:,:,:] likelihood, double[:,:,:] probs, double lambdaParam, double mu, double[:] N, double[:] D, double[:] prev):
    cdef int nrows=probs.shape[0]
    cdef int ncols=probs.shape[1]
    cdef int nclasses=probs.shape[2]
    cdef double retVal
    retVal=iterateMarginals(&likelihood[0,0,0], &probs[0,0,0], nrows, ncols, nclasses, lambdaParam, mu, &N[0], &D[0], &prev[0])
    return retVal
    
cpdef optimize_marginals(double[:,:,:] likelihood, double[:,:,:] probs, double lambdaParam, double mu, int maxIter, double tolerance):
    cdef int nrows=probs.shape[0]
    cdef int ncols=probs.shape[1]
    cdef int nclasses=probs.shape[2]
    cdef double retVal
    retVal=optimizeMarginals(&likelihood[0,0,0], &probs[0,0,0], nrows, ncols, nclasses, lambdaParam, mu, maxIter, tolerance)
    return retVal

cpdef compute_neg_log_likelihood_constant_models(double[:,:] img, double[:] means, double[:] variances, double[:,:,:] negLogLikelihood):
    cdef int nrows=negLogLikelihood.shape[0]
    cdef int ncols=negLogLikelihood.shape[1]
    cdef int nclasses=negLogLikelihood.shape[2]
    cdef int retVal
    retVal=computeNegLogLikelihoodConstantModels(&img[0,0], nrows, ncols, nclasses, &means[0], &variances[0], &negLogLikelihood[0,0,0])
    return retVal

cpdef initialize_constant_models(double[:,:] img, int nclasses):
    cdef int nrows=img.shape[0]
    cdef int ncols=img.shape[1]
    cdef int retVal
    cdef double[:] means=np.zeros((nclasses,))
    cdef double[:] variances=np.zeros((nclasses,))
    retVal=initializeConstantModels(&img[0,0], nrows, ncols, nclasses, &means[0], &variances[0])
    return means, variances
    
cpdef initialize_normalized_likelihood(double[:,:,:] negLogLikelihood, double[:,:,:] probs):
    cdef int nrows=probs.shape[0]
    cdef int ncols=probs.shape[1]
    cdef int nclasses=probs.shape[2]
    cdef int retVal
    retVal=initializeNormalizedLikelihood(&negLogLikelihood[0,0,0], nrows, ncols, nclasses, &probs[0,0,0])

cpdef initialize_maximum_likelihood(double[:,:,:] negLogLikelihood, double[:,:,:] probs):
    cdef int nrows=probs.shape[0]
    cdef int ncols=probs.shape[1]
    cdef int nclasses=probs.shape[2]
    cdef int retVal
    retVal=initializeMaximumLikelihoodProbs(&negLogLikelihood[0,0,0], nrows, ncols, nclasses, &probs[0,0,0])

cpdef ecqmmf(double[:,:] img, int nclasses, double lambdaParam, double mu, int outerIter, int innerIter, double tolerance):
    cdef int iter_count
    cdef int nrows=img.shape[0]
    cdef int ncols=img.shape[1]
    cdef double[:] N=np.zeros((nclasses, ))
    cdef double[:] D=np.zeros((nclasses, ))
    cdef double[:] prev=np.zeros((nclasses, ))
    cdef double[:] means=np.zeros((nclasses,))
    cdef double[:] variances=np.zeros((nclasses,))
    cdef double [:,:,:] probs=np.zeros((nrows, ncols, nclasses))
    cdef double [:,:,:] negLogLikelihood=np.zeros((nrows, ncols, nclasses))
    cdef double [:,:] segmented=np.zeros((nrows, ncols))
    cdef double mseProbs, mseModels
    cdef int retVal, inner, outer
    retVal=initializeConstantModels(&img[0,0], nrows, ncols, nclasses, &means[0], &variances[0])
    computeNegLogLikelihoodConstantModels(&img[0,0], nrows, ncols, nclasses, &means[0], &variances[0], &negLogLikelihood[0,0,0])
    #initializeMaximumLikelihoodProbs(&negLogLikelihood[0,0,0], nrows, ncols, nclasses, &probs[0,0,0])
    initializeNormalizedLikelihood(&negLogLikelihood[0,0,0], nrows, ncols, nclasses, &probs[0,0,0])
    for outer in range(outerIter):
        for inner in range(innerIter):
            mseProbs=iterateMarginals(&negLogLikelihood[0,0,0], &probs[0,0,0], nrows, ncols, nclasses, lambdaParam, mu, &N[0], &D[0], &prev[0])
            print '\tInner:',inner,'/',innerIter,'. Max mse:',mseProbs,'. Tol:',tolerance
            if(mseProbs<tolerance):
                break
        mseModels=updateConstantModels(&img[0,0], &probs[0,0,0], nrows, ncols, nclasses, &means[0], &variances[0])
        print 'Outer:',outer,'/',outerIter, '. MSE models:', mseModels
        if(mseModels<tolerance):
            break
        retVal=computeNegLogLikelihoodConstantModels(&img[0,0], nrows, ncols, nclasses, &means[0], &variances[0], &negLogLikelihood[0,0,0])
    getImageModes(&probs[0,0,0], nrows, ncols, nclasses, &means[0], &segmented[0,0])
    return segmented, means, variances, probs
    
    
    