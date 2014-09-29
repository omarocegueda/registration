# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:40:28 2013

@author: khayyam
"""
import numpy as np
import scipy as sp
from scipy import ndimage
import nibabel as nib
import os
import tensorFieldUtils as tf

# def affine_registration(static, moving):
#     from nipy.io.files import nipy2nifti, nifti2nipy
#     from nipy.algorithms.registration import HistogramRegistration, resample
#     nipy_static = nifti2nipy(static)
#     nipy_moving = nifti2nipy(moving)
#     similarity = 'crl1' #'crl1' 'cc', 'mi', 'nmi', 'cr', 'slr'
#     interp = 'tri' #'pv', 'tri',
#     renormalize = True
#     optimizer = 'powell'
#     R = HistogramRegistration(nipy_static, nipy_moving, similarity=similarity,
#                           interp=interp, renormalize=renormalize)
#     T = R.optimize('affine', optimizer=optimizer)
#     warped= resample(nipy_moving, T, reference=nipy_static, interp_order=1)
#     warped = nipy2nifti(warped, strict=True)
#     return warped, T

# warped, affine_init = affine_registration(nib_static, nib_moving)

def getBaseFileName(fname):
    base=os.path.basename(fname)
    noExt=os.path.splitext(base)[0]
    while(noExt!=base):
        base=noExt
        noExt=os.path.splitext(base)[0]
    return noExt    
    

def getDistribution(img1, img2):
    sh=img1.shape
    dist=np.zeros((256,256))
    for i in range(sh[0]):
        for j in range(sh[1]):
            a=int(img1[i,j])
            b=int(img2[i,j])
            dist[a,b]+=1
    return dist

def createDeformationField2D_type1(nrows, ncols, maxDistp):
    deff=np.ndarray((nrows, ncols, 2), dtype=np.float64)
    midCol=ncols//2
    midRow=nrows//2
    for i in range(nrows):
        deff[i,:,0]=maxDistp*np.sin(2*np.pi*(np.array(range(ncols), dtype=np.float64)-midCol)/ncols)
    for j in range(ncols):
        deff[:,j,1]=maxDistp*np.sin(2*np.pi*(np.array(range(nrows), dtype=np.float64)-midRow)/nrows)
    v=np.array(range(nrows), dtype=np.float64)-midRow
    h=np.array(range(ncols), dtype=np.float64)-midCol
    nrm=midRow**2+midCol**2
    p=np.exp(-(v[:,None]**2+h[None,:]**2)/(0.1*nrm))
    p=(p-p.min())/(p.max()-p.min())
    deff[:,:,0]*=p
    deff[:,:,1]*=p
    return deff

def createDeformationField2D_type2(nrows, ncols, maxDistp):
    deff=np.ndarray((nrows, ncols, 2), dtype=np.float64)
    midCol=ncols//2
    midRow=nrows//2
    for i in range(nrows):
        deff[i,:,0]=maxDistp*np.sin(2*np.pi*(np.array(range(ncols), dtype=np.float64)-midCol)/ncols)
        deff[i,:,1]=maxDistp*np.sin(2*np.pi*(np.array(range(ncols), dtype=np.float64)-midCol)/ncols)
    for j in range(ncols):
        deff[:,j,0]*=np.sin(2*np.pi*(np.array(range(nrows), dtype=np.float64)-midRow)/nrows)
        deff[:,j,1]*=np.sin(2*np.pi*(np.array(range(nrows), dtype=np.float64)-midRow)/nrows)
    return deff

def createDeformationField2D_type3(nrows, ncols, maxDistp):
    deff=np.ndarray((nrows, ncols, 2), dtype=np.float64)
    X0,X1=np.mgrid[0:nrows, 0:ncols]
    midCol=ncols//2
    midRow=nrows//2
    nn=np.sqrt(midCol*midCol+midRow*midRow)
    factor=maxDistp/nn
    deff[...,0]=(X1-midCol)*(-factor)
    deff[...,1]=(X0-midRow)*(factor)
    return deff


def createDeformationField3D_type2(dims, maxDistp):
    deff=np.ndarray(dims+(3,), dtype=np.float64)
    dims=np.array(dims, dtype=np.int32)
    mid=dims//2
    factor=maxDistp**(1.0/3.0)
    for i in range(dims[0]):
        deff[i,:,:,0]=factor*np.sin(2*np.pi*(np.array(range(dims[0]), dtype=np.float64)-mid[0])/dims[0])
        deff[i,:,:,1]=factor*np.sin(2*np.pi*(np.array(range(dims[0]), dtype=np.float64)-mid[0])/dims[0])
        deff[i,:,:,2]=factor*np.sin(2*np.pi*(np.array(range(dims[0]), dtype=np.float64)-mid[0])/dims[0])
    for j in range(dims[1]):
        deff[:,j,:,0]*=factor*np.sin(2*np.pi*(np.array(range(dims[1]), dtype=np.float64)-mid[1])/dims[1])
        deff[:,j,:,1]*=factor*np.sin(2*np.pi*(np.array(range(dims[1]), dtype=np.float64)-mid[1])/dims[1])
        deff[:,j,:,2]*=factor*np.sin(2*np.pi*(np.array(range(dims[1]), dtype=np.float64)-mid[1])/dims[1])
    for k in range(dims[2]):
        deff[:,:,k,0]*=factor*np.sin(2*np.pi*(np.array(range(dims[2]), dtype=np.float64)-mid[2])/dims[2])
        deff[:,:,k,1]*=factor*np.sin(2*np.pi*(np.array(range(dims[2]), dtype=np.float64)-mid[2])/dims[2])
        deff[:,:,k,2]*=factor*np.sin(2*np.pi*(np.array(range(dims[2]), dtype=np.float64)-mid[2])/dims[2])
    return deff


def getRotationMatrix(angles):
    ca=np.cos(angles[0])
    cb=np.cos(angles[1])
    cg=np.cos(angles[2])
    sa=np.sin(angles[0])
    sb=np.sin(angles[1])
    sg=np.sin(angles[2])
    return np.array([[cb*cg,-ca*sg+sa*sb*cg,sa*sg+ca*sb*cg],[cb*sg,ca*cg+sa*sb*sg,-sa*cg+ca*sb*sg],[-sb,sa*cb,ca*cb]])


def getRotationMatrix2D(angle):
    c=np.cos(angle)
    s=np.sin(angle)
    return np.array([[c, -s],[s, c]])


def computeJacobianField(displacement):
    g00,g01=sp.gradient(displacement[...,0])
    g10,g11=sp.gradient(displacement[...,1])
    return (1+g00)*(1+g11)-g10*g01


def readAntsAffine(fname):
    '''
    readAntsAffine('IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt')
    '''
    try:
        with open(fname) as f:
            lines=[line.strip() for line in f.readlines()]
    except IOError:
        print 'Can not open file: ', fname
        return
    if not (lines[0]=="#Insight Transform File V1.0"):
        print 'Unknown file format'
        return
    if lines[1]!="#Transform 0":
        print 'Unknown transformation type'
        return
    A=np.zeros((3,3))
    b=np.zeros((3,))
    c=np.zeros((3,))
    for line in lines[2:]:
        data=line.split()
        if data[0]=='Transform:':
            if data[1]!='MatrixOffsetTransformBase_double_3_3' and data[1]!='AffineTransform_double_3_3':
                print 'Unknown transformation type'
                return
        elif data[0]=='Parameters:':
            parameters=np.array([float(s) for s in data[1:]], dtype=np.float64)
            A=parameters[:9].reshape((3,3))
            b=parameters[9:]
        elif data[0]=='FixedParameters:':
            c=np.array([float(s) for s in data[1:]], dtype=np.float64)
    T=np.ndarray(shape=(4,4), dtype=np.float64)
    T[:3,:3]=A[...]
    T[3,:]=0
    T[3,3]=1
    T[:3,3]=b+c-A.dot(c)
    ############This conversion is necessary for compatibility between itk and nibabel#########
    conversion=np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    T=conversion.dot(T.dot(conversion))
    ###########################################################################################
    return T
