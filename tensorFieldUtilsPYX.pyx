# -*- coding: utf-8 -*-
#distutils: language = c++
#distutils: sources = tensorFieldUtilsCPP.cpp
"""
Created on Thu Sep 19 15:38:56 2013

@author: khayyam
"""
from cython.view cimport memoryview
from cython.view cimport array as cvarray
from cpython cimport bool
import numpy as np
import scipy as sp

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
    double computeDemonsStep2D(double *deltaField, double *gradientField, int *dims, double maxStepSize, double scale, double *demonsStep)
    double iterateDisplacementField2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField, double *residual)
    double computeEnergySSD2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField)
    double iterateDisplacementField3DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField, double *residual)
    double iterateResidualDisplacementFieldSSD2D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField)
    int computeResidualDisplacementFieldSSD2D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField, double *residual)
    double iterateResidualDisplacementFieldSSD3D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField)
    int computeResidualDisplacementFieldSSD3D(double *deltaField, double *sigmaField, double *gradientField, double *target, int *dims, double lambdaParam, double *displacementField, double *residual)
    double computeEnergySSD3DCPP(double *deltaField, double *sigmaField, double *gradientField, int *dims, double lambdaParam, double *displacementField)
    void computeMaskedVolumeClassStatsProbsCPP(int *mask, double *img, int *dims, int numLabels, double *probs, double *means, double *variances)
    void integrateMaskedWeightedTensorFieldProductsProbsCPP(int *mask, double *q, int *dims, double *diff, int nclasses, double *probs, double *weights, double *Aw, double *bw)
    double iterateMaskedDisplacementField2DCPP(double *deltaField, double *sigmaField, double *gradientField, int *mask, int *dims, double lambdaParam, double *displacementField, double *residual)
    int invertVectorField(double *d, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *invd, double *stats)
    int invertVectorFieldFixedPoint(double *d, int nrows, int ncols, int maxIter, double tolerance, double *invd, double *start, double *stats)
    int invertVectorFieldFixedPoint3D(double *d, int nslices, int nrows, int ncols, int maxIter, double tolerance, double *invd, double *start, double *stats)
    int composeVectorFields(double *d1, double *d2, int nrows, int ncols, double *comp, double *stats)
    int vectorFieldExponential(double *v, int nrows, int ncols, double *expv, double *invexpv)
    int readDoubleBuffer(char *fname, int nDoubles, double *buffer)
    int writeDoubleBuffer(double *buffer, int nDoubles, char *fname)
    void createInvertibleDisplacementField(int nrows, int ncols, double b, double m, double *dField)
    int invertVectorFieldYan(double *forward, int nrows, int ncols, int maxloop, double tolerance, double *inv)
    void countSupportingDataPerPixel(double *forward, int nrows, int ncols, int *counts)
    int vectorFieldAdjointInterpolation(double *d1, double *d2, int nrows, int ncols, double *sol)
    int vectorFieldInterpolation(double *d1, double *d2, int nrows, int ncols, double *comp)
    int invertVectorField_TV_L2(double *forward, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *inv)
    void consecutiveLabelMap(int *v, int n, int *out)
    int composeVectorFields3D(double *d1, double *d2, int nslices, int nrows, int ncols, double *comp, double *stats)
    int vectorFieldExponential3D(double *v, int nslices, int nrows, int ncols, double *expv, double *invexpv)
    int upsampleDisplacementField(double *d1, int nrows, int ncols, double *up, int nr, int nc)
    int upsampleDisplacementField3D(double *d1, int ns, int nr, int nc, double *up, int nslices, int nrows, int ncols)
    int downsampleDisplacementField(double *d1, int nr, int nc, double *down)
    int downsampleScalarField(double *d1, int nr, int nc, double *down)
    int downsampleDisplacementField3D(double *d1, int ns, int nr, int nc, double *down)
    int downsampleScalarField3D(double *d1, int ns, int nr, int nc, double *down)
    int warpImageAffine(double *img, int nrImg, int ncImg, double *affine, double *warped, int nrRef, int ncRef)
    int warpImage(double *img, int nrImg, int ncImg, double *d1, int nrows, int ncols, double *affinePre, double *affinePost, double *warped)
    int warpImageNN(double *img, int nrImg, int ncImg, double *d1, int nrows, int ncols, double *affinePre, double *affinePost, double *warped)
    int warpDiscreteImageNNAffine(int *img, int nrImg, int ncImg, double *affine, int *warped, int nrRef, int ncRef)
    int warpDiscreteImageNN(int *img, int nrImg, int ncImg, double *d1, int nrows, int ncols, double *affinePre, double *affinePost, int *warped)
    int warpVolumeAffine(double *volume, int nsVol, int nrVol, int ncVol, double *affine, double *warped, int nsRef, int nrRef, int ncRef)
    int multVectorFieldByAffine3D(double *displacement, int nslices, int nrows, int ncols, double *affine)
    int warpVolume(double *volume, int nsVol, int nrVol, int ncVol, double *d1, int nslices, int nrows, int ncols, double *affinePre, double *affinePost, double *warped)
    int warpVolumeNN(double *volume, int nsVol, int nrVol, int ncVol, double *d1, int nslices, int nrows, int ncols, double *affinePre, double *affinePost, double *warped)
    int warpDiscreteVolumeNNAffine(int *volume, int nsVol, int nrVol, int ncVol, double *affine, int *warped, int nsRef, int nrRef, int ncRef)
    int warpDiscreteVolumeNN(int *volume, int nsVol, int nrVol, int ncVol, double *d1, int nslices, int nrows, int ncols, double *affinePre, double *affinePost, int *warped)
    int invertVectorField3D(double *forward, int nslices, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *inv, double *stats)
    int prependAffineToDisplacementField(double *d1, int nslices, int nrows, int ncols, double *affine)
    int apendAffineToDisplacementField(double *d1, int nslices, int nrows, int ncols, double *affine)
    void getVotingSegmentation(int *votes, int nslices, int nrows, int ncols, int nvotes, int *seg)
    int getDisplacementRange(double *d, int nslices, int nrows, int ncols, double *affine, double *minVal, double *maxVal)
    int computeJacard(int *A, int *B, int nslices, int nrows, int ncols, double *jacard, int nlabels)

cdef checkFortran(a):
    pass
#    if np.isfortran(np.array(a)):
#        print 'Warning: passing fortran array to C++ routine. C-order is internally assummed when processing multi-dimensional arrays in C++.'
def consecutive_label_map(int[:,:,:] v):
    cdef int n=v.shape[0]*v.shape[1]*v.shape[2]
    cdef int[:,:,:] out=cvarray(shape=(v.shape[0], v.shape[1], v.shape[2]), itemsize=sizeof(int), format="i")
    consecutiveLabelMap(&v[0,0,0], n, &out[0,0,0])
    return out

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
    checkFortran(q)
    checkFortran(diff)
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
    checkFortran(v)
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
    checkFortran(v)
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
    checkFortran(v)
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
    checkFortran(v)
    quantizeVolumeCPP(&v[0,0,0], &dims[0], numLevels, &out[0,0,0], &levels[0], &hist[0])
    return out, levels, hist

cpdef computeImageClassStatsCYTHON(double[:,:] v, int numLabels, int[:,:] labels):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    cdef double[:] means=np.zeros(shape=(numLabels,), dtype=np.double)
    cdef double[:] variances=np.zeros(shape=(numLabels, ), dtype=np.double)
    checkFortran(v)
    checkFortran(labels)
    computeImageClassStatsCPP(&v[0,0], &dims[0], numLabels, &labels[0,0], &means[0], &variances[0])
    return means, variances

cpdef computeMaskedImageClassStatsCYTHON(int[:,:] mask, double[:,:] v, int numLabels, int[:,:] labels):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    cdef double[:] means=np.zeros(shape=(numLabels,), dtype=np.double)
    cdef double[:] variances=np.zeros(shape=(numLabels, ), dtype=np.double)
    checkFortran(mask)
    checkFortran(v)
    checkFortran(labels)
    computeMaskedImageClassStatsCPP(&mask[0,0], &v[0,0], &dims[0], numLabels, &labels[0,0], &means[0], &variances[0])
    return means, variances

cpdef computeVolumeClassStatsCYTHON(double[:,:,:] v, int numLabels, int[:,:,:] labels):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    dims[2]=v.shape[2]
    cdef double[:] means=np.zeros(shape=(numLabels,), dtype=np.double)
    cdef double[:] variances=np.zeros(shape=(numLabels, ), dtype=np.double)
    checkFortran(v)
    checkFortran(labels)
    computeVolumeClassStatsCPP(&v[0,0,0], &dims[0], numLabels, &labels[0,0,0], &means[0], &variances[0])
    return means, variances

cpdef computeMaskedVolumeClassStatsCYTHON(int[:,:,:] mask, double[:,:,:] v, int numLabels, int[:,:,:] labels):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=v.shape[0]
    dims[1]=v.shape[1]
    dims[2]=v.shape[2]
    cdef double[:] means=np.zeros(shape=(numLabels,), dtype=np.double)
    cdef double[:] variances=np.zeros(shape=(numLabels, ), dtype=np.double)
    checkFortran(mask)
    checkFortran(v)
    checkFortran(labels)
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
    checkFortran(q)
    checkFortran(diff)
    checkFortran(labels)
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
    checkFortran(mask)
    checkFortran(q)
    checkFortran(diff)
    checkFortran(labels)
    integrateMaskedWeightedTensorFieldProductsCPP(&mask[0,0,0], &q[0,0,0,0], &dims[0], &diff[0,0,0], numLabels, &labels[0,0,0], &weights[0], &Aw[0,0], &bw[0])
    return Aw,bw

cpdef compute_demons_step2D(double[:,:] deltaField, double[:,:,:] gradientField,  double maxStepSize, double scale):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    cdef double[:,:,:] demonsStep=np.empty(shape=(dims[0], dims[1], 2), dtype=np.double)
    cdef double maxNorm
    maxNorm=computeDemonsStep2D(&deltaField[0,0], &gradientField[0,0,0], &dims[0], maxStepSize, scale, &demonsStep[0,0,0])
    return demonsStep

cpdef compute_demons_step3D(double[:,:,:] deltaField, double[:,:,:,:] gradientField,  double maxStepSize, double scale):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    dims[2]=deltaField.shape[2]
    cdef double[:,:,:,:] demonsStep=np.empty(shape=(dims[0], dims[1], dims[2], 3), dtype=np.double)
    cdef double maxNorm
    maxNorm=computeDemonsStep2D(&deltaField[0,0,0], &gradientField[0,0,0,0], &dims[0], maxStepSize, scale, &demonsStep[0,0,0,0])
    return demonsStep

cpdef iterateDisplacementField2DCYTHON(double[:,:] deltaField, double[:,:] sigmaField, double[:,:,:] gradientField,  double lambdaParam, double[:,:,:] displacementField, double[:,:] residuals):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    cdef double *sigmaFieldPointer=NULL 
    if sigmaField!=None:
        sigmaFieldPointer=&sigmaField[0,0]
    cdef double *residualsPointer=NULL 
    if residuals!=None:
        residualsPointer=&residuals[0,0]
    maxDisplacement=iterateDisplacementField2DCPP(&deltaField[0,0], sigmaFieldPointer, &gradientField[0,0,0], &dims[0], lambdaParam, &displacementField[0,0,0], residualsPointer)
    return maxDisplacement

cpdef iterate_residual_displacement_field_SSD2D(double[:,:] deltaField, double[:,:] sigmaField, double[:,:,:] gradientField,  double[:,:,:] target, double lambdaParam, double[:,:,:] displacementField):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    cdef double retVal
    cdef double *targetPointer=NULL
    if target!=None:
        targetPointer=&target[0,0,0]
    retVal=iterateResidualDisplacementFieldSSD2D(&deltaField[0,0], &sigmaField[0,0], &gradientField[0,0,0], targetPointer, &dims[0], lambdaParam, &displacementField[0,0,0]);

cpdef compute_residual_displacement_field_SSD3D(double[:,:,:] deltaField, double[:,:,:] sigmaField, double[:,:,:,:] gradientField,  double[:,:,:,:] target, double lambdaParam, double[:,:,:,:] displacementField, double[:,:,:,:] residual):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    dims[2]=deltaField.shape[2]
    cdef int retVal
    cdef double *targetPointer=NULL
    if target!=None:
        targetPointer=&target[0,0,0,0]
    if residual==None:
        residual=np.empty(shape=(dims[0], dims[1], dims[2], 3), dtype=np.double)
    retVal=computeResidualDisplacementFieldSSD3D(&deltaField[0,0,0], &sigmaField[0,0,0], &gradientField[0,0,0,0], targetPointer, &dims[0], lambdaParam, &displacementField[0,0,0,0], &residual[0,0,0,0]);    
    return residual

cpdef iterate_residual_displacement_field_SSD3D(double[:,:,:] deltaField, double[:,:,:] sigmaField, double[:,:,:,:] gradientField,  double[:,:,:,:] target, double lambdaParam, double[:,:,:,:] displacementField):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    dims[2]=deltaField.shape[2]
    cdef double retVal
    cdef double *targetPointer=NULL
    if target!=None:
        targetPointer=&target[0,0,0,0]
    retVal=iterateResidualDisplacementFieldSSD3D(&deltaField[0,0,0], &sigmaField[0,0,0], &gradientField[0,0,0,0], targetPointer, &dims[0], lambdaParam, &displacementField[0,0,0,0]);

cpdef compute_residual_displacement_field_SSD2D(double[:,:] deltaField, double[:,:] sigmaField, double[:,:,:] gradientField,  double[:,:,:] target, double lambdaParam, double[:,:,:] displacementField, double[:,:,:] residual):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    cdef int retVal
    cdef double *targetPointer=NULL
    if target!=None:
        targetPointer=&target[0,0,0]
    if residual==None:
        residual=np.empty(shape=(dims[0], dims[1], 2), dtype=np.double)
    retVal=computeResidualDisplacementFieldSSD2D(&deltaField[0,0], &sigmaField[0,0], &gradientField[0,0,0], targetPointer, &dims[0], lambdaParam, &displacementField[0,0,0], &residual[0,0,0]);    
    return residual

cpdef compute_energy_SSD2D(double[:,:] deltaField, double[:,:] sigmaField, double[:,:,:] gradientField,  double lambdaParam, double[:,:,:] displacementField):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    cdef double *sigmaFieldPointer=NULL 
    cdef double energy
    if sigmaField!=None:
        sigmaFieldPointer=&sigmaField[0,0]
    energy=computeEnergySSD2DCPP(&deltaField[0,0], sigmaFieldPointer, &gradientField[0,0,0], &dims[0], lambdaParam, &displacementField[0,0,0])
    return energy


cpdef iterateMaskedDisplacementField2DCYTHON(double[:,:] deltaField, double[:,:] sigmaField, double[:,:,:] gradientField,  int[:,:] mask, double lambdaParam, double[:,:,:] displacementField, double[:,:] residuals):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    checkFortran(deltaField)
    checkFortran(sigmaField)
    checkFortran(gradientField)
    checkFortran(mask)
    checkFortran(displacementField)
    checkFortran(residuals)
    if residuals==None:
        maxDisplacement=iterateMaskedDisplacementField2DCPP(&deltaField[0,0], &sigmaField[0,0], &gradientField[0,0,0], &mask[0,0], &dims[0], lambdaParam, &displacementField[0,0,0], NULL)
    else:
        maxDisplacement=iterateMaskedDisplacementField2DCPP(&deltaField[0,0], &sigmaField[0,0], &gradientField[0,0,0], &mask[0,0], &dims[0], lambdaParam, &displacementField[0,0,0], &residuals[0,0])
    return maxDisplacement

cpdef iterateDisplacementField3DCYTHON(double[:,:,:] deltaField, double[:,:,:] sigmaField, double[:,:,:,:] gradientField,  double lambdaParam, double[:,:,:,:] displacementField, double[:,:,:] residuals):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    dims[2]=deltaField.shape[2]
    checkFortran(deltaField)
    checkFortran(sigmaField)
    checkFortran(gradientField)
    checkFortran(displacementField)
    checkFortran(residuals)
    if residuals==None:
        maxDisplacement=iterateDisplacementField3DCPP(&deltaField[0,0,0], &sigmaField[0,0,0], &gradientField[0,0,0,0], &dims[0], lambdaParam, &displacementField[0,0,0,0], NULL)
    else:
        maxDisplacement=iterateDisplacementField3DCPP(&deltaField[0,0,0], &sigmaField[0,0,0], &gradientField[0,0,0,0], &dims[0], lambdaParam, &displacementField[0,0,0,0], &residuals[0,0,0])
    return maxDisplacement

cpdef compute_energy_SSD3D(double[:,:,:] deltaField, double[:,:,:] sigmaField, double[:,:,:,:] gradientField,  double lambdaParam, double[:,:,:,:] displacementField):
    cdef int[:] dims=cvarray(shape=(3,), itemsize=sizeof(int), format="i")
    dims[0]=deltaField.shape[0]
    dims[1]=deltaField.shape[1]
    dims[2]=deltaField.shape[2]
    cdef double energy
    energy=computeEnergySSD3DCPP(&deltaField[0,0,0], &sigmaField[0,0,0], &gradientField[0,0,0,0], &dims[0], lambdaParam, &displacementField[0,0,0,0])
    return energy


cpdef computeMaskedVolumeClassStatsProbsCYTHON(int[:,:] mask, double[:,:] img, double[:,:,:] probs):
    cdef int[:] dims=cvarray(shape=(2,), itemsize=sizeof(int), format="i")
    dims[0]=probs.shape[0]
    dims[1]=probs.shape[1]
    nclasses=probs.shape[2]
    cdef double[:] means=np.zeros(shape=(nclasses,), dtype=np.double)
    cdef double[:] variances=np.zeros(shape=(nclasses, ), dtype=np.double)
    checkFortran(mask)
    checkFortran(img)
    checkFortran(probs)
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
    checkFortran(mask)
    checkFortran(q)
    checkFortran(diff)
    checkFortran(probs)
    integrateMaskedWeightedTensorFieldProductsProbsCPP(&mask[0,0], &q[0,0,0], &dims[0], &diff[0,0], nclasses, &probs[0,0,0], &weights[0], &Aw[0,0], &bw[0])
    return Aw,bw

cpdef invert_vector_field(double[:,:,:] d, double lambdaParam, int maxIter, double tolerance):
    cdef int retVal
    cdef int nrows=d.shape[0]
    cdef int ncols=d.shape[1]
    cdef double[:] stats=cvarray(shape=(2,), itemsize=sizeof(double), format='d')
    cdef double[:,:,:] invd=np.zeros_like(d)
    checkFortran(d)
    retVal=invertVectorField(&d[0,0,0], nrows, ncols, lambdaParam, maxIter, tolerance, &invd[0,0,0], &stats[0])
    #print 'MSE:', stats[0], 'Last iteration:', int(stats[1])
    return invd

cpdef invert_vector_field_tv_l2(double[:,:,:] d, double lambdaParam, int maxIter, double tolerance):
    cdef int retVal
    cdef int nrows=d.shape[0]
    cdef int ncols=d.shape[1]
    cdef double[:,:,:] invd=np.zeros_like(d)
    checkFortran(d)
    retVal=invertVectorField_TV_L2(&d[0,0,0], nrows, ncols, lambdaParam, maxIter, tolerance, &invd[0,0,0]);
    return invd


cpdef invert_vector_field_Yan(double[:,:,:] d, int maxIter, double tolerance):
    cdef int retVal
    cdef int nrows=d.shape[0]
    cdef int ncols=d.shape[1]
    cdef double[:,:,:] invd=np.zeros_like(d)
    checkFortran(d)
    retVal=invertVectorFieldYan(&d[0,0,0], nrows, ncols, maxIter, tolerance, &invd[0,0,0])
    return invd

cpdef invert_vector_field_fixed_point(double[:,:,:] d, int maxIter, double tolerance, double[:,:,:] start=None):
    cdef int retVal
    cdef int nrows=d.shape[0]
    cdef int ncols=d.shape[1]
    cdef double[:] stats=cvarray(shape=(2,), itemsize=sizeof(double), format='d')
    cdef double[:,:,:] invd=np.zeros_like(d)
    cdef double *startPointer=NULL
    if start!=None:
        startPointer=&start[0,0,0]
    retVal=invertVectorFieldFixedPoint(&d[0,0,0], nrows, ncols, maxIter, tolerance, &invd[0,0,0], startPointer, &stats[0])
    #print 'MSE:', stats[0], 'Last iteration:', int(stats[1])
    return invd

cpdef compose_vector_fields(double[:,:,:] d1, double[:,:,:] d2):
    cdef int nrows=d1.shape[0]
    cdef int ncols=d1.shape[1]
    cdef double[:,:,:] comp=np.zeros_like(d1)
    cdef int retVal
    cdef double[:] stats=cvarray(shape=(3,), itemsize=sizeof(double), format='d')
    checkFortran(d1)
    checkFortran(d2)
    retVal=composeVectorFields(&d1[0,0,0], &d2[0,0,0], nrows, ncols, &comp[0,0,0], &stats[0])
    #print 'Max displacement:', stats[0], 'Mean displacement:', stats[1], '(', stats[2], ')'
    return comp, stats

cpdef compose_vector_fields3D(double[:,:,:,:] d1, double[:,:,:,:] d2):
    cdef int nslices=d1.shape[0]
    cdef int nrows=d1.shape[1]
    cdef int ncols=d1.shape[2]
    cdef double[:,:,:,:] comp=np.zeros_like(d1)
    cdef double[:] stats=cvarray(shape=(3,), itemsize=sizeof(double), format='d')
    cdef int retVal
    checkFortran(d1)
    checkFortran(d2)    
    retVal=composeVectorFields3D(&d1[0,0,0,0], &d2[0,0,0,0], nslices, nrows, ncols, &comp[0,0,0,0], &stats[0])
    #print 'Max displacement:', stats[0], 'Mean displacement:', stats[1], '(', stats[2], ')'
    return comp, stats

cpdef vector_field_interpolation(double[:,:,:] d1, double[:,:,:] d2):
    cdef int nrows=d1.shape[0]
    cdef int ncols=d1.shape[1]
    cdef double[:,:,:] sol=np.zeros_like(d1)
    cdef int retVal
    checkFortran(d1)
    checkFortran(d2)
    retVal=vectorFieldInterpolation(&d1[0,0,0], &d2[0,0,0], nrows, ncols, &sol[0,0,0])
    return sol

cpdef vector_field_adjoint_interpolation(double[:,:,:] d1, double[:,:,:] d2):
    cdef int nrows=d1.shape[0]
    cdef int ncols=d1.shape[1]
    cdef double[:,:,:] sol=np.zeros_like(d1)
    cdef int retVal
    checkFortran(d1)
    checkFortran(d2)
    retVal=vectorFieldAdjointInterpolation(&d1[0,0,0], &d2[0,0,0], nrows, ncols, &sol[0,0,0])
    return sol

cpdef vector_field_exponential(double[:,:,:] v, bool computeInverse):
    cdef double[:,:,:] expv = np.zeros_like(v)
    cdef double[:,:,:] invexpv = None
    cdef int retVal
    cdef int nrows=v.shape[0]
    cdef int ncols=v.shape[1]
    checkFortran(v)
    if computeInverse:
        invexpv = np.zeros_like(v)
        retVal=vectorFieldExponential(&v[0,0,0], nrows, ncols, &expv[0,0,0], &invexpv[0,0,0])
    else:
        retVal=vectorFieldExponential(&v[0,0,0], nrows, ncols, &expv[0,0,0], NULL)
    return expv, invexpv

cpdef vector_field_exponential3D(double[:,:,:,:] v, bool computeInverse):
    cdef double[:,:,:,:] expv = np.zeros_like(v)
    cdef double[:,:,:,:] invexpv=None
    cdef int retVal
    cdef int nslices=v.shape[0]
    cdef int nrows=v.shape[1]
    cdef int ncols=v.shape[2]
    checkFortran(v)
    if(computeInverse):
        invexpv = np.zeros_like(v)
        retVal=vectorFieldExponential3D(&v[0,0,0,0], nslices, nrows, ncols, &expv[0,0,0,0], &invexpv[0,0,0,0])
    else:
        retVal=vectorFieldExponential3D(&v[0,0,0,0], nslices, nrows, ncols, &expv[0,0,0,0], NULL)
    return expv, invexpv

cpdef read_double_buffer(bytes fname, int nDoubles):
    cdef double[:] buff=np.zeros(shape=(nDoubles,))
    readDoubleBuffer(fname, nDoubles, &buff[0])
    return buff

cpdef write_double_buffer(double[:] buff, bytes fname):
    cdef int nDoubles=buff.shape[0]
    writeDoubleBuffer(&buff[0], nDoubles, fname)

cpdef create_invertible_displacement_field(int nrows, int ncols, double b, double m):
    '''
        import tensorFieldUtils as tf
        GT=tf.create_invertible_displacement_field(256, 256, 0.2, 8)
    '''
    cdef double[:,:,:] dField = np.ndarray((nrows, ncols,2), dtype=np.float64)
    createInvertibleDisplacementField(nrows, ncols, b, m, &dField[0,0,0])
    return dField

cpdef count_supporting_data_per_pixel(double[:,:,:] forward):
    cdef int nrows=forward.shape[0]
    cdef int ncols=forward.shape[1]
    cdef int[:,:] counts=np.zeros(shape=(nrows, ncols), dtype=np.int32)
    checkFortran(forward)
    countSupportingDataPerPixel(&forward[0,0,0], nrows, ncols, &counts[0,0])
    return counts

def downsample_scalar_field(double[:,:] field):
    cdef int nr=field.shape[0]
    cdef int nc=field.shape[1]
    cdef double[:,:] down = np.ndarray(((nr+1)//2, (nc+1)//2), dtype=np.float64)
    downsampleScalarField(&field[0,0], nr, nc, &down[0,0]);
    return down

def downsample_displacement_field(double[:,:,:] field):
    cdef int nr=field.shape[0]
    cdef int nc=field.shape[1]
    cdef double[:,:,:] down = np.ndarray(((nr+1)//2, (nc+1)//2,2), dtype=np.float64)
    downsampleDisplacementField(&field[0,0,0], nr, nc, &down[0,0,0]);
    return down

def downsample_scalar_field3D(double[:,:,:] field):
    cdef int ns=field.shape[0]
    cdef int nr=field.shape[1]
    cdef int nc=field.shape[2]
    cdef double[:,:,:] down = np.ndarray(((ns+1)//2, (nr+1)//2, (nc+1)//2), dtype=np.float64)
    downsampleScalarField3D(&field[0,0,0], ns, nr, nc, &down[0,0,0]);
    return down

def downsample_displacement_field3D(double[:,:,:,:] field):
    cdef int ns=field.shape[0]
    cdef int nr=field.shape[1]
    cdef int nc=field.shape[2]
    cdef double[:,:,:,:] down = np.ndarray(((ns+1)//2, (nr+1)//2, (nc+1)//2,3), dtype=np.float64)
    downsampleDisplacementField3D(&field[0,0,0,0], ns, nr, nc, &down[0,0,0,0]);
    return down

def upsample_displacement_field(double[:,:,:] field, int[:] targetShape):
    cdef int nr=field.shape[0]
    cdef int nc=field.shape[1]
    cdef int nrows=targetShape[0]
    cdef int ncols=targetShape[1]
    checkFortran(field)
    cdef double[:,:,:] up = np.ndarray((nrows, ncols,2), dtype=np.float64)
    upsampleDisplacementField(&field[0,0,0], nr, nc, &up[0,0,0],nrows, ncols);
    return up

def upsample_displacement_field3D(double[:,:,:,:] field, int[:] targetShape):
    cdef int ns=field.shape[0]
    cdef int nr=field.shape[1]
    cdef int nc=field.shape[2]
    cdef int nslices=targetShape[0]
    cdef int nrows=targetShape[1]
    cdef int ncols=targetShape[2]
    checkFortran(field)
    cdef double[:,:,:,:] up = np.ndarray((nslices, nrows, ncols,3), dtype=np.float64)
    upsampleDisplacementField3D(&field[0,0,0,0], ns, nr, nc, &up[0,0,0,0],nslices, nrows, ncols);
    return up

def warp_image_affine(double[:,:] img, int[:]refShape, double[:,:] affine):
    cdef int nrImg=img.shape[0]
    cdef int ncImg=img.shape[1]
    cdef double[:,:] warped = np.ndarray((refShape[0], refShape[1]), dtype=np.float64)
    checkFortran(img)
    checkFortran(affine)
    warpImageAffine(&img[0,0], nrImg, ncImg, &affine[0,0], &warped[0,0], refShape[0], refShape[1])
    return warped

def warp_image(double[:,:] img, double[:,:,:] displacement, double[:,:] affinePre=None, double[:,:] affinePost=None):
    cdef int nrImg=img.shape[0]
    cdef int ncImg=img.shape[1]
    cdef int nrows=img.shape[0]
    cdef int ncols=img.shape[1]
    cdef double *displacementPointer=NULL
    cdef double *affinePrePointer=NULL
    cdef double *affinePostPointer=NULL
    cdef double[:,:] warped = np.ndarray((nrows, ncols), dtype=np.float64)
    if displacement!=None:
        displacementPointer=&displacement[0,0,0]
        nrows=displacement.shape[0]
        ncols=displacement.shape[1]
    if affinePre!=None:
        affinePrePointer=&affinePre[0,0]
    if affinePost!=None:
        affinePostPointer=&affinePost[0,0]
    warpImage(&img[0,0], nrImg, ncImg, displacementPointer, nrows, ncols, affinePrePointer, affinePostPointer, &warped[0,0]);
    return warped

def warp_imageNN(double[:,:] img, double[:,:,:] displacement, double[:,:] affinePre=None, double[:,:] affinePost=None):
    cdef int nrows=img.shape[0]
    cdef int ncols=img.shape[1]
    cdef int nrImg=img.shape[0]
    cdef int ncImg=img.shape[1]
    cdef double *displacementPointer=NULL
    cdef double *affinePrePointer=NULL
    cdef double *affinePostPointer=NULL
    cdef double[:,:] warped = np.ndarray((nrows, ncols), dtype=np.float64)
    if displacement!=None:
        displacementPointer=&displacement[0,0,0]
        nrows=displacement.shape[0]
        ncols=displacement.shape[1]
    if affinePre!=None:
        affinePrePointer=&affinePre[0,0]
    if affinePost!=None:
        affinePostPointer=&affinePost[0,0]
    warpImageNN(&img[0,0], nrImg, ncImg, displacementPointer, nrows, ncols, affinePrePointer, affinePostPointer, &warped[0,0])
    return warped

def warp_discrete_imageNNAffine(int[:,:] img, int[:] refShape, double[:,:] affine=None):
    cdef int nrImg=img.shape[0]
    cdef int ncImg=img.shape[1]
    cdef int[:,:] warped = np.ndarray((refShape[0], refShape[1]), dtype=np.int32)
    checkFortran(img)
    checkFortran(affine)
    warpDiscreteImageNNAffine(&img[0,0], nrImg, ncImg, &affine[0,0], &warped[0,0], refShape[0], refShape[1])
    return warped

def warp_discrete_imageNN(int[:,:] img, double[:,:,:] displacement, double[:,:] affinePre=None, double[:,:] affinePost=None):
    cdef int nrows=img.shape[0]
    cdef int ncols=img.shape[1]
    cdef int nrImg=img.shape[0]
    cdef int ncImg=img.shape[1]
    cdef double *displacementPointer=NULL
    cdef double *affinePrePointer=NULL
    cdef double *affinePostPointer=NULL
    cdef int[:,:] warped = np.ndarray((nrows, ncols), dtype=np.int32)
    if displacement!=None:
        displacementPointer=&displacement[0,0,0]
        nrows=displacement.shape[0]
        ncols=displacement.shape[1]
    if affinePre!=None:
        affinePrePointer=&affinePre[0,0]
    if affinePost!=None:
        affinePostPointer=&affinePost[0,0]
    warpDiscreteImageNN(&img[0,0], nrImg, ncImg, displacementPointer, nrows, ncols, affinePrePointer, affinePostPointer, &warped[0,0])
    return warped

def warp_volume_affine(double[:,:,:] volume, int[:]refShape, double[:,:] affine=None):
    cdef int nsVol=volume.shape[0]
    cdef int nrVol=volume.shape[1]
    cdef int ncVol=volume.shape[2]
    cdef double[:,:,:] warped = np.ndarray((refShape[0], refShape[1], refShape[2]), dtype=np.float64)
    checkFortran(volume)
    checkFortran(affine)
    if affine==None:
        warpVolumeAffine(&volume[0,0,0], nsVol, nrVol, ncVol, NULL, &warped[0,0,0], refShape[0], refShape[1], refShape[2])
    else:
        warpVolumeAffine(&volume[0,0,0], nsVol, nrVol, ncVol, &affine[0,0], &warped[0,0,0], refShape[0], refShape[1], refShape[2])
    return warped

def mult_vector_field_by_affine3D(double[:,:,:,:] displacement, double[:,:] affine=None):
    cdef int ns=displacement.shape[0]
    cdef int nr=displacement.shape[1]
    cdef int nc=displacement.shape[2]
    multVectorFieldByAffine3D(&displacement[0,0,0,0], ns, nr, nc, &affine[0,0])

def warp_volume(double[:,:,:] volume, double[:,:,:,:] displacement, double[:,:] affinePre=None, double[:,:] affinePost=None):
    cdef int nslices=volume.shape[0]
    cdef int nrows=volume.shape[1]
    cdef int ncols=volume.shape[2]
    cdef int nsVol=volume.shape[0]
    cdef int nrVol=volume.shape[1]
    cdef int ncVol=volume.shape[2]
    cdef double *displacementPointer=NULL
    cdef double *affinePrePointer=NULL
    cdef double *affinePostPointer=NULL
    cdef double[:,:,:] warped = np.ndarray((nslices, nrows, ncols), dtype=np.float64)
    if displacement!=None:
        displacementPointer=&displacement[0,0,0,0]
        nslices=displacement.shape[0]
        nrows=displacement.shape[1]
        ncols=displacement.shape[2]
    if affinePre!=None:
        affinePrePointer=&affinePre[0,0]
    if affinePost!=None:
        affinePostPointer=&affinePost[0,0]
    warpVolume(&volume[0,0,0], nsVol, nrVol, ncVol, displacementPointer, nslices, nrows, ncols, affinePrePointer, affinePostPointer, &warped[0,0,0])
    return warped

def warp_volumeNN(double[:,:,:] volume, double[:,:,:,:] displacement, double[:,:] affinePre=None, double[:,:] affinePost=None):
    cdef int nslices=volume.shape[0]
    cdef int nrows=volume.shape[1]
    cdef int ncols=volume.shape[2]
    cdef int nsVol=volume.shape[0]
    cdef int nrVol=volume.shape[1]
    cdef int ncVol=volume.shape[2]
    cdef double *displacementPointer=NULL
    cdef double *affinePrePointer=NULL
    cdef double *affinePostPointer=NULL
    cdef double[:,:,:] warped = np.ndarray((nslices, nrows, ncols), dtype=np.float64)
    if displacement!=None:
        displacementPointer=&displacement[0,0,0,0]
        nslices=displacement.shape[0]
        nrows=displacement.shape[1]
        ncols=displacement.shape[2]
    if affinePre!=None:
        affinePrePointer=&affinePre[0,0]
    if affinePost!=None:
        affinePostPointer=&affinePost[0,0]
    warpVolumeNN(&volume[0,0,0], nsVol, nrVol, ncVol, displacementPointer, nslices, nrows, ncols, affinePrePointer, affinePostPointer, &warped[0,0,0])
    return warped

def warp_discrete_volumeNNAffine(int[:,:,:] volume, int[:] refShape, double[:,:] affine=None):
    cdef int nsVol=volume.shape[0]
    cdef int nrVol=volume.shape[1]
    cdef int ncVol=volume.shape[2]
    cdef int[:,:,:] warped = np.ndarray((refShape[0], refShape[1], refShape[2]), dtype=np.int32)
    checkFortran(volume)
    checkFortran(affine)
    if affine==None:
        warpDiscreteVolumeNNAffine(&volume[0,0,0], nsVol, nrVol, ncVol, NULL, &warped[0,0,0], refShape[0], refShape[1], refShape[2])
    else:
        warpDiscreteVolumeNNAffine(&volume[0,0,0], nsVol, nrVol, ncVol, &affine[0,0], &warped[0,0,0], refShape[0], refShape[1], refShape[2])
    return warped

def warp_discrete_volumeNN(int[:,:,:] volume, double[:,:,:,:] displacement, double[:,:] affinePre=None, double[:,:] affinePost=None):
    cdef int nslices=volume.shape[0]
    cdef int nrows=volume.shape[1]
    cdef int ncols=volume.shape[2]
    cdef int nsVol=volume.shape[0]
    cdef int nrVol=volume.shape[1]
    cdef int ncVol=volume.shape[2]
    cdef double *displacementPointer=NULL
    cdef double *affinePrePointer=NULL
    cdef double *affinePostPointer=NULL
    cdef int[:,:,:] warped = np.ndarray((nslices, nrows, ncols), dtype=np.int32)
    if displacement!=None:
        displacementPointer=&displacement[0,0,0,0]
        nslices=displacement.shape[0]
        nrows=displacement.shape[1]
        ncols=displacement.shape[2]
    if affinePre!=None:
        affinePrePointer=&affinePre[0,0]
    if affinePost!=None:
        affinePostPointer=&affinePost[0,0]
    warpDiscreteVolumeNN(&volume[0,0,0], nsVol, nrVol, ncVol, displacementPointer, nslices, nrows, ncols, affinePrePointer, affinePostPointer, &warped[0,0,0])
    return warped

def get_voting_segmentation(int[:,:,:,:] votes):
    cdef int nslices=votes.shape[0]
    cdef int nrows=votes.shape[1]
    cdef int ncols=votes.shape[2]
    cdef int nvotes=votes.shape[3]
    cdef int[:,:,:] seg = np.ndarray((nslices, nrows, ncols), dtype=np.int32)
    checkFortran(votes)
    getVotingSegmentation(&votes[0,0,0,0], nslices, nrows, ncols, nvotes, &seg[0,0,0])
    return seg

def invert_vector_field3D(double[:,:,:,:] d, double lambdaParam, int maxIter, double tolerance):
    cdef int retVal
    cdef int nslices=d.shape[0]
    cdef int nrows=d.shape[1]
    cdef int ncols=d.shape[2]
    cdef double[:] stats=cvarray(shape=(2,), itemsize=sizeof(double), format='d')
    cdef double[:,:,:,:] invd=np.zeros_like(d)
    checkFortran(d)
    retVal=invertVectorField3D(&d[0,0,0,0], nslices, nrows, ncols, lambdaParam, maxIter, tolerance, &invd[0,0,0,0], &stats[0])
    #print 'MSE:', stats[0], 'Last iteration:', int(stats[1])
    return invd

def invert_vector_field_fixed_point3D(double[:,:,:,:] d, int maxIter, double tolerance, double[:,:,:,:] start=None):
    cdef int retVal
    cdef int nslices=d.shape[0]
    cdef int nrows=d.shape[1]
    cdef int ncols=d.shape[2]
    cdef double[:] stats=cvarray(shape=(2,), itemsize=sizeof(double), format='d')
    cdef double[:,:,:,:] invd=np.zeros_like(d)
    checkFortran(d)
    cdef double *startPointer=NULL
    if start!=None:
        startPointer=&start[0,0,0,0]
    retVal=invertVectorFieldFixedPoint3D(&d[0,0,0,0], nslices, nrows, ncols, maxIter, tolerance, &invd[0,0,0,0], startPointer, &stats[0])
    #print 'MSE:', stats[0], 'Last iteration:', int(stats[1])
    return invd

def prepend_affine_to_displacement_field(double[:,:,:,:] d, double[:,:] affine):
    cdef int retVal
    cdef int nslices=d.shape[0]
    cdef int nrows=d.shape[1]
    cdef int ncols=d.shape[2]
    retVal=prependAffineToDisplacementField(&d[0,0,0,0], nslices, nrows, ncols, &affine[0,0])
    
def apend_affine_to_displacement_field(double[:,:,:,:] d, double[:,:] affine):
    cdef int retVal
    cdef int nslices=d.shape[0]
    cdef int nrows=d.shape[1]
    cdef int ncols=d.shape[2]
    retVal=apendAffineToDisplacementField(&d[0,0,0,0], nslices, nrows, ncols, &affine[0,0])

def get_displacement_range(double[:,:,:,:] d, double[:,:] affine):
    cdef int retVal
    cdef int nslices=d.shape[0]
    cdef int nrows=d.shape[1]
    cdef int ncols=d.shape[2]
    cdef double[:] minVal = np.ndarray((3,), dtype=np.float64)
    cdef double[:] maxVal = np.ndarray((3,), dtype=np.float64)
    if affine==None:
        retVal=getDisplacementRange(&d[0,0,0,0], nslices, nrows, ncols, NULL, &minVal[0], &maxVal[0])
    else:
        retVal=getDisplacementRange(&d[0,0,0,0], nslices, nrows, ncols, &affine[0,0], &minVal[0], &maxVal[0])
    return minVal, maxVal

def compute_jacard(int[:,:,:] A, int[:,:,:] B, int nlabels):
    cdef int retVal
    cdef int nslices=A.shape[0]
    cdef int nrows=A.shape[1]
    cdef int ncols=A.shape[2]
    cdef double[:] jacard = np.ndarray((nlabels, ), dtype=np.float64)
    retVal=computeJacard(&A[0,0,0], &B[0,0,0], nslices, nrows, ncols, &jacard[0], nlabels)
    return jacard
