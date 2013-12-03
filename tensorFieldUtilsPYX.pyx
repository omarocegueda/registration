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
    int invertVectorFieldFixedPoint(double *d, int nrows, int ncols, int maxIter, double tolerance, double *invd, double *stats)
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
    int composeVectorFields3D(double *d1, double *d2, int nslices, int nrows, int ncols, double *comp)
    int vectorFieldExponential3D(double *v, int nslices, int nrows, int ncols, double *expv, double *invexpv)
    int upsampleDisplacementField3D(double *d1, int ns, int nr, int nc, double *up, int nslices, int nrows, int ncols)
    int warpVolume(double *volume, double *d1, int nslices, int nrows, int ncols, double *warped)
    int warpVolumeNN(double *volume, double *d1, int nslices, int nrows, int ncols, double *warped)
    int warpDiscreteVolumeNN(int *volume, double *d1, int nslices, int nrows, int ncols, int *warped)
    int invertVectorField3D(double *forward, int nslices, int nrows, int ncols, double lambdaParam, int maxIter, double tolerance, double *inv, double *stats)
    void getVotingSegmentation(int *votes, int nslices, int nrows, int ncols, int nvotes, int *seg)

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

cpdef invert_vector_field_tv_l2(double[:,:,:] d, double lambdaParam, int maxIter, double tolerance):
    cdef int retVal
    cdef int nrows=d.shape[0]
    cdef int ncols=d.shape[1]
    cdef double[:,:,:] invd=np.zeros_like(d)
    retVal=invertVectorField_TV_L2(&d[0,0,0], nrows, ncols, lambdaParam, maxIter, tolerance, &invd[0,0,0]);
    return invd


cpdef invert_vector_field_Yan(double[:,:,:] d, int maxIter, double tolerance):
    cdef int retVal
    cdef int nrows=d.shape[0]
    cdef int ncols=d.shape[1]
    cdef double[:,:,:] invd=np.zeros_like(d)
    retVal=invertVectorFieldYan(&d[0,0,0], nrows, ncols, maxIter, tolerance, &invd[0,0,0])
    return invd

#cpdef invert_vector_field_fixed_point(double[:,:,:] d, int maxIter, double tolerance):
#    cdef double[:,:,:] invd=np.zeros_like(d)
#    sh=d.shape
#    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]
#    for it in range(maxIter):
#        invd[:,:,0], invd[:,:,1]=(-1*ndimage.map_coordinates(d[:,:,0], [X0+invd[...,0], X1+invd[...,1]], prefilter=const_prefilter_map_coordinates),
#                                    -1*ndimage.map_coordinates(d[:,:,1], [X0+invd[...,0], X1+invd[...,1]], prefilter=const_prefilter_map_coordinates))
#    return invd
#    cdef int retVal
#    cdef int nrows=d.shape[0]
#    cdef int ncols=d.shape[1]
#    cdef double[:] stats=cvarray(shape=(2,), itemsize=sizeof(double), format='d')
#    cdef double[:,:,:] invd=np.zeros_like(d)
#    retVal=invertVectorFieldFixedPoint(&d[0,0,0], nrows, ncols, maxIter, tolerance, &invd[0,0,0], &stats[0])
#    print 'Max GS step:', stats[0], 'Last iteration:', int(stats[1])
#    return invd

cpdef compose_vector_fields(double[:,:,:] d1, double[:,:,:] d2):
    cdef int nrows=d1.shape[0]
    cdef int ncols=d1.shape[1]
    cdef double[:,:,:] comp=np.zeros_like(d1)
    cdef int retVal
    cdef double[:] stats=cvarray(shape=(3,), itemsize=sizeof(double), format='d')
    retVal=composeVectorFields(&d1[0,0,0], &d2[0,0,0], nrows, ncols, &comp[0,0,0], &stats[0])
    #print 'Max displacement:', stats[0], 'Mean displacement:', stats[1], '(', stats[2], ')'
    return comp, stats

cpdef compose_vector_fields3D(double[:,:,:,:] d1, double[:,:,:,:] d2):
    cdef int nslices=d1.shape[0]
    cdef int nrows=d1.shape[1]
    cdef int ncols=d1.shape[2]
    cdef double[:,:,:,:] comp=np.zeros_like(d1)
    cdef int retVal
    retVal=composeVectorFields3D(&d1[0,0,0,0], &d2[0,0,0,0], nslices, nrows, ncols, &comp[0,0,0,0])
    #print 'Max displacement:', stats[0], 'Mean displacement:', stats[1], '(', stats[2], ')'
    return comp

cpdef vector_field_interpolation(double[:,:,:] d1, double[:,:,:] d2):
    cdef int nrows=d1.shape[0]
    cdef int ncols=d1.shape[1]
    cdef double[:,:,:] sol=np.zeros_like(d1)
    cdef int retVal
    retVal=vectorFieldInterpolation(&d1[0,0,0], &d2[0,0,0], nrows, ncols, &sol[0,0,0])
    return sol

cpdef vector_field_adjoint_interpolation(double[:,:,:] d1, double[:,:,:] d2):
    cdef int nrows=d1.shape[0]
    cdef int ncols=d1.shape[1]
    cdef double[:,:,:] sol=np.zeros_like(d1)
    cdef int retVal
    retVal=vectorFieldAdjointInterpolation(&d1[0,0,0], &d2[0,0,0], nrows, ncols, &sol[0,0,0])
    return sol

cpdef vector_field_exponential(double[:,:,:] v):
    cdef double[:,:,:] expv = np.zeros_like(v)
    cdef double[:,:,:] invexpv = np.zeros_like(v)
    cdef int retVal
    cdef int nrows=v.shape[0]
    cdef int ncols=v.shape[1]
    retVal=vectorFieldExponential(&v[0,0,0], nrows, ncols, &expv[0,0,0], &invexpv[0,0,0])
    return expv, invexpv

cpdef vector_field_exponential3D(double[:,:,:,:] v, bool computeInverse):
    cdef double[:,:,:,:] expv = np.zeros_like(v)
    cdef double[:,:,:,:] invexpv=None
    if(computeInverse):
        invexpv = np.zeros_like(v)
    cdef int retVal
    cdef int nslices=v.shape[0]
    cdef int nrows=v.shape[1]
    cdef int ncols=v.shape[2]
    if(computeInverse):
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
    countSupportingDataPerPixel(&forward[0,0,0], nrows, ncols, &counts[0,0])
    return counts

def upsample_displacement_field3D(double[:,:,:,:] field, int[:] targetShape):
    cdef int ns=field.shape[0]
    cdef int nr=field.shape[1]
    cdef int nc=field.shape[2]
    cdef int nslices=targetShape[0]
    cdef int nrows=targetShape[1]
    cdef int ncols=targetShape[2]
    cdef double[:,:,:,:] up = np.ndarray((nslices, nrows, ncols,3), dtype=np.float64)
    upsampleDisplacementField3D(&field[0,0,0,0], ns, nr, nc, &up[0,0,0,0],nslices, nrows, ncols);
    return up

def warp_volume(double[:,:,:] volume, double[:,:,:,:] displacement):
    cdef int nslices=volume.shape[0]
    cdef int nrows=volume.shape[1]
    cdef int ncols=volume.shape[2]
    cdef double[:,:,:] warped = np.ndarray((nslices, nrows, ncols), dtype=np.float64)
    warpVolume(&volume[0,0,0], &displacement[0,0,0,0], nslices, nrows, ncols, &warped[0,0,0]);
    return warped

def warp_volumeNN(double[:,:,:] volume, double[:,:,:,:] displacement):
    cdef int nslices=volume.shape[0]
    cdef int nrows=volume.shape[1]
    cdef int ncols=volume.shape[2]
    cdef double[:,:,:] warped = np.ndarray((nslices, nrows, ncols), dtype=np.float64)
    warpVolumeNN(&volume[0,0,0], &displacement[0,0,0,0], nslices, nrows, ncols, &warped[0,0,0]);
    return warped

def warp_discrete_volumeNN(int[:,:,:] volume, double[:,:,:,:] displacement):
    cdef int nslices=volume.shape[0]
    cdef int nrows=volume.shape[1]
    cdef int ncols=volume.shape[2]
    cdef int[:,:,:] warped = np.ndarray((nslices, nrows, ncols), dtype=np.int32)
    warpDiscreteVolumeNN(&volume[0,0,0], &displacement[0,0,0,0], nslices, nrows, ncols, &warped[0,0,0]);
    return warped

def get_voting_segmentation(int[:,:,:,:] votes):
    cdef int nslices=votes.shape[0]
    cdef int nrows=votes.shape[1]
    cdef int ncols=votes.shape[2]
    cdef int nvotes=votes.shape[3]
    cdef int[:,:,:] seg = np.ndarray((nslices, nrows, ncols), dtype=np.int32)
    getVotingSegmentation(&votes[0,0,0,0], nslices, nrows, ncols, nvotes, &seg[0,0,0])
    return seg


def invert_vector_field3D(double[:,:,:,:] d, double lambdaParam, int maxIter, double tolerance):
    cdef int retVal
    cdef int nslices=d.shape[0]
    cdef int nrows=d.shape[1]
    cdef int ncols=d.shape[2]
    cdef double[:] stats=cvarray(shape=(2,), itemsize=sizeof(double), format='d')
    cdef double[:,:,:,:] invd=np.zeros_like(d)
    retVal=invertVectorField3D(&d[0,0,0,0], nslices, nrows, ncols, lambdaParam, maxIter, tolerance, &invd[0,0,0,0], &stats[0])
    print 'Max step:', stats[0], 'Last iteration:', int(stats[1])
    return invd
