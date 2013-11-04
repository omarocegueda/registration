# -*- coding: utf-8 -*-
#distutils: language = c++
#distutils: sources = tensorFieldUtilsCPP.cpp
"""
Created on Thu Sep 19 15:38:56 2013

@author: khayyam
"""
from cython.view cimport memoryview
from cython.view cimport array as cvarray
import numpy as np

cdef extern from "tensorFieldUtilsCPP.h":
    void integrateTensorFieldProductsCPP(double *q, int *dims, double *diff, double *A, double *b)
    void quantizeImageCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist)
    void quantizePositiveImageCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist)
    void quantizeVolumeCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist)
    void quantizePositiveVolumeCPP(double *v, int *dims, int numLevels, int *out, double *levels, int *hist)
    void computeImageClassStatsCPP(double *v, int *dims, int numLabels, int *labels, double *means, double *variances)
    void computeMaskedImageClassStatsCPP(int *mask, double *v, int *dims, int numLabels, int *labels, double *means, double *variances)
    void computeVolumeClassStatsCPP(double *v, int *dims, int numLabels, int *labels, double *means, double *variances)
    void computeMaskedVolumeClassStatsCPP(int *masked, double *v, int *dims, int numLabels, int *labels, double *means, double *variances)
    void integrateWeightedTensorFieldProductsCPP(double *q, int *dims, double *diff, int numLabels, int *labels, double *weights, double *Aw, double *bw)
    void integrateMaskedWeightedTensorFieldProductsCPP(int *mask, double *q, int *dims, double *diff, int numLabels, int *labels, double *weights, double *Aw, double *bw)
    double iterateDisplacementField2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *previousDisplacement, double *displacementField, double *residual)
    double iterateDisplacementField3DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *previousDisplacement, double *displacementField, double *residual)
    void computeMaskedVolumeClassStatsProbsCPP(int *mask, double *img, int *dims, int numLabels, double *probs, double *means, double *variances)
    void integrateMaskedWeightedTensorFieldProductsProbsCPP(int *mask, double *q, int *dims, double *diff, int nclasses, double *probs, double *weights, double *Aw, double *bw)
    double iterateMaskedDisplacementField2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *mask, int *dims, double lambdaParam, double *previousDisplacement, double *displacementField, double *residual)
    int invertVectorField(double *d, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *invd, double *stats)
    int composeVectorFields(double *d1, double *d2, int nrows, int ncols, double *comp, double *stats)
    int vectorFieldExponential(double *v, int nrows, int ncols, double *expv, double *invexpv)

def testFunction(param):
    print 'Testing', param

cpdef integrateTensorFieldProductsCYTHON(double[:, :, :, :] q, double[:, :, :] diff):
    cdef int[:] dims=cvarray(shape=(4,), itemsize=sizeof(int), format="i")
    dims[0]=q.shape[0]
    dims[1]=q.shape[1]
    dims[2]=q.shape[2]
    dims[3]=q.shape[3]
    cdef int k=dims[3]    
    cdef double[:,:] A=np.zeros(shape=(k, k), dtype=np.double)
    cdef double[:] b=np.zeros(shape=(k,), dtype=np.double)
    integrateTensorFieldProductsCPP(&q[0,0,0,0], &dims[0], &diff[0,0,0], &A[0,0], &b[0])
    return A,b

cpdef quantizeImageCYTHON(double[:,:]v, int numLevels):
    '''
    out, levels, hist=quantizeVolumeCYTHON(v, numLevels)
    '''
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    cdef int[:,:] out=np.zeros(shape=(dims[0], dims[1],), dtype=np.int32)
    cdef double[:] levels=np.zeros(shape=(numLevels,), dtype=np.double)
    cdef int[:] hist=np.zeros(shape=(numLevels,), dtype=np.int32)
    quantizeImageCPP(&v[0,0], &dims[0], numLevels, &out[0,0], &levels[0], &hist[0])
    return out, levels, hist

cpdef quantizePositiveImageCYTHON(double[:,:]v, int numLevels):
    '''
    out, levels, hist=quantizeVolumeCYTHON(v, numLevels)
    '''
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    cdef int[:,:] out=np.zeros(shape=(dims[0], dims[1],), dtype=np.int32)
    cdef double[:] levels=np.zeros(shape=(numLevels,), dtype=np.double)
    cdef int[:] hist=np.zeros(shape=(numLevels,), dtype=np.int32)
    quantizePositiveImageCPP(&v[0,0], &dims[0], numLevels, &out[0,0], &levels[0], &hist[0])
    return out, levels, hist

cpdef quantizePositiveVolumeCYTHON(double[:,:,:]v, int numLevels):
    '''
    out, levels, hist=quantizeVolumeCYTHON(v, numLevels)
    '''
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    dims[2]=v.shape[2]
    cdef int[:,:,:] out=np.zeros(shape=(dims[0], dims[1], dims[2]), dtype=np.int32)
    cdef double[:] levels=np.zeros(shape=(numLevels,), dtype=np.double)
    cdef int[:] hist=np.zeros(shape=(numLevels,), dtype=np.int32)
    quantizePositiveVolumeCPP(&v[0,0,0], &dims[0], numLevels, &out[0,0,0], &levels[0], &hist[0])
    return out, levels, hist

cpdef quantizeVolumeCYTHON(double[:,:,:]v, int numLevels):
    '''
    out, levels, hist=quantizeVolumeCYTHON(v, numLevels)
    '''
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    dims[2]=v.shape[2]
    cdef int[:,:,:] out=np.zeros(shape=(dims[0], dims[1], dims[2],), dtype=np.int32)
    cdef double[:] levels=np.zeros(shape=(numLevels,), dtype=np.double)
    cdef int[:] hist=np.zeros(shape=(numLevels,), dtype=np.int32)
    quantizeVolumeCPP(&v[0,0,0], &dims[0], numLevels, &out[0,0,0], &levels[0], &hist[0])
    return out, levels, hist

cpdef computeImageClassStatsCYTHON(double[:,:] v, int numLabels, int[:,:] labels):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    cdef double[:] means=np.zeros(shape=(numLabels,), dtype=np.double)
    cdef double[:] variances=np.zeros(shape=(numLabels, ), dtype=np.double)
    computeImageClassStatsCPP(&v[0,0], &dims[0], numLabels, &labels[0,0], &means[0], &variances[0])
    return means, variances

cpdef computeMaskedImageClassStatsCYTHON(int[:,:] mask, double[:,:] v, int numLabels, int[:,:] labels):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    cdef double[:] means=np.zeros(shape=(numLabels,), dtype=np.double)
    cdef double[:] variances=np.zeros(shape=(numLabels, ), dtype=np.double)
    computeMaskedImageClassStatsCPP(&mask[0,0], &v[0,0], &dims[0], numLabels, &labels[0,0], &means[0], &variances[0])
    return means, variances

cpdef computeVolumeClassStatsCYTHON(double[:,:,:] v, int numLabels, int[:,:,:] labels):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    dims[2]=v.shape[2]
    cdef double[:] means=np.zeros(shape=(numLabels,), dtype=np.double)
    cdef double[:] variances=np.zeros(shape=(numLabels, ), dtype=np.double)
    computeVolumeClassStatsCPP(&v[0,0,0], &dims[0], numLabels, &labels[0,0,0], &means[0], &variances[0])
    return means, variances

cpdef computeMaskedVolumeClassStatsCYTHON(int[:,:,:] mask, double[:,:,:] v, int numLabels, int[:,:,:] labels):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    dims[2]=v.shape[2]
    cdef double[:] means=np.zeros(shape=(numLabels,), dtype=np.double)
    cdef double[:] variances=np.zeros(shape=(numLabels, ), dtype=np.double)
    computeMaskedVolumeClassStatsCPP(&mask[0,0,0], &v[0,0,0], &dims[0], numLabels, &labels[0,0,0], &means[0], &variances[0])
    return means, variances

cpdef integrateWeightedTensorFieldProductsCYTHON(double[:,:,:,:] q, double[:,:,:] diff, int numLabels, int[:,:,:] labels, double[:] weights):
    cdef int[:] dims=cvarray(shape=(4,), itemsize=sizeof(int), format="i")
    dims[0]=q.shape[0]
    dims[1]=q.shape[1]
    dims[2]=q.shape[2]
    dims[3]=q.shape[3]
    cdef int k=dims[3]
    cdef double[:,:] Aw=np.zeros(shape=(k, k), dtype=np.double)
    cdef double[:] bw=np.zeros(shape=(k,), dtype=np.double)
    integrateWeightedTensorFieldProductsCPP(&q[0,0,0,0], &dims[0], &diff[0,0,0], numLabels, &labels[0,0,0], &weights[0], &Aw[0,0], &bw[0])
    return Aw,bw

cpdef integrateMaskedWeightedTensorFieldProductsCYTHON(int[:,:,:] mask, double[:,:,:,:] q, double[:,:,:] diff, int numLabels, int[:,:,:] labels, double[:] weights):
    cdef int[:] dims=cvarray(shape=(4,), itemsize=sizeof(int), format="i")
    dims[0]=q.shape[0]
    dims[1]=q.shape[1]
    dims[2]=q.shape[2]
    dims[3]=q.shape[3]
    cdef int k=dims[3]
    cdef double[:,:] Aw=np.zeros(shape=(k, k), dtype=np.double)
    cdef double[:] bw=np.zeros(shape=(k,), dtype=np.double)
    integrateMaskedWeightedTensorFieldProductsCPP(&mask[0,0,0], &q[0,0,0,0], &dims[0], &diff[0,0,0], numLabels, &labels[0,0,0], &weights[0], &Aw[0,0], &bw[0])
    return Aw,bw

cpdef iterateDisplacementField2DCYTHON(double[:,:] deltaField, double[:,:] sigmaField, double[:,:,:] gradientField,  double lambdaParam, double[:,:,:] previousDisplacement, double[:,:,:] displacementField, double[:,:] residuals):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    maxDisplacement=iterateDisplacementField2DCPP(&deltaField[0,0], &sigmaField[0,0], &gradientField[0,0,0], &dims[0], lambdaParam, &previousDisplacement[0,0,0], &displacementField[0,0,0], &residuals[0,0])
    return maxDisplacement

cpdef iterateMaskedDisplacementField2DCYTHON(double[:,:] deltaField, double[:,:] sigmaField, double[:,:,:] gradientField,  int[:,:] mask, double lambdaParam, double[:,:,:] previousDisplacement, double[:,:,:] displacementField, double[:,:] residuals):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    maxDisplacement=iterateMaskedDisplacementField2DCPP(&deltaField[0,0], &sigmaField[0,0], &gradientField[0,0,0], &mask[0,0], &dims[0], lambdaParam, &previousDisplacement[0,0,0], &displacementField[0,0,0], &residuals[0,0])
    return maxDisplacement

cpdef iterateDisplacementField3DCYTHON(double[:,:,:] deltaField, double[:,:,:] sigmaField, double[:,:,:,:] gradientField,  double lambdaParam, double[:,:,:,:] previousDisplacement, double[:,:,:,:] displacementField, double[:,:,:] residuals):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    dims[2]=deltaField.shape[2]
    maxDisplacement=iterateDisplacementField3DCPP(&deltaField[0,0,0], &sigmaField[0,0,0], &gradientField[0,0,0,0], &dims[0], lambdaParam, &previousDisplacement[0,0,0,0], &displacementField[0,0,0,0], &residuals[0,0,0])
    return maxDisplacement

cpdef computeMaskedVolumeClassStatsProbsCYTHON(int[:,:] mask, double[:,:] img, double[:,:,:] probs):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=probs.shape[0]
    dims[1]=probs.shape[1]
    nclasses=probs.shape[2]
    cdef double[:] means=np.zeros(shape=(nclasses,), dtype=np.double)
    cdef double[:] variances=np.zeros(shape=(nclasses, ), dtype=np.double)
    computeMaskedVolumeClassStatsProbsCPP(&mask[0,0], &img[0,0], &dims[0], nclasses, &probs[0,0,0], &means[0], &variances[0])
    return means, variances
    
cpdef integrateMaskedWeightedTensorFieldProductsProbsCYTHON(int[:,:] mask, double[:,:,:] q, double[:,:] diff, int nclasses, double[:,:,:] probs, double[:] weights):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=q.shape[0]
    dims[1]=q.shape[1]
    dims[2]=q.shape[2]
    cdef int k=dims[2]
    cdef double[:,:] Aw=np.zeros(shape=(k, k), dtype=np.double)
    cdef double[:] bw=np.zeros(shape=(k,), dtype=np.double)
    integrateMaskedWeightedTensorFieldProductsProbsCPP(&mask[0,0], &q[0,0,0], &dims[0], &diff[0,0], nclasses, &probs[0,0,0], &weights[0], &Aw[0,0], &bw[0])
    return Aw,bw

cpdef invert_vector_field(double[:,:,:] d, double lambdaParam, int maxIter, double tolerance):
    cdef int retVal
    cdef int nrows=d.shape[0]
    cdef int ncols=d.shape[1]
    cdef double[:] stats=cvarray(shape=(2,), itemsize=sizeof(double), format='d')
    cdef double[:,:,:] invd=np.zeros_like(d)
    retVal=invertVectorField(&d[0,0,0], nrows, ncols, lambdaParam, maxIter, tolerance, &invd[0,0,0], &stats[0])
    print 'Max GS step:', stats[0], 'Last iteration:', int(stats[1])
    return invd

cpdef compose_vector_fields(double[:,:,:] d1, double[:,:,:] d2):
    cdef int nrows=d1.shape[0]
    cdef int ncols=d1.shape[1]
    cdef double[:,:,:] comp=np.zeros_like(d1)
    cdef int retVal
    cdef double[:] stats=cvarray(shape=(3,), itemsize=sizeof(double), format='d')
    retVal=composeVectorFields(&d1[0,0,0], &d2[0,0,0], nrows, ncols, &comp[0,0,0], &stats[0])
    print 'Max displacement:', stats[0], 'Mean displacement:', stats[1], '(', stats[2], ')'
    return comp

cpdef vector_field_exponential(double[:,:,:] v):
    cdef double[:,:,:] expv = np.zeros_like(v)
    cdef double[:,:,:] invexpv = np.zeros_like(v)
    cdef int retVal
    cdef int nrows=v.shape[0]
    cdef int ncols=v.shape[1]
    retVal=vectorFieldExponential(&v[0,0,0], nrows, ncols, &expv[0,0,0], &invexpv[0,0,0])
    return expv, invexpv


