# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:51:32 2013

@author: khayyam
"""
import numpy as np
import scipy as sp
import tensorFieldUtils as tf
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
import registrationCommon as rcommon
from registrationCommon import const_prefilter_map_coordinates

###############################################################
####### Non-linear Monomodal registration - EM (2D)############
###############################################################

def estimateNewMonomodalDeformationField2D(left, right, lambdaParam, previousDisplacement=None):
    epsilon=1e-4
    sh=left.shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]
    displacement     =np.zeros(shape=(left.shape)+(2,), dtype=np.float64)
    residuals=np.zeros(shape=(left.shape), dtype=np.float64)
    gradientField    =np.empty(shape=(left.shape)+(2,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(left.shape)+(2,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
    warped=ndimage.map_coordinates(left, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1]], prefilter=const_prefilter_map_coordinates)
    sigmaField=np.ones_like(warped, dtype=np.float64)
    deltaField=right-warped
    g0, g1=sp.gradient(warped)
    gradientField[:,:,0]=g0
    gradientField[:,:,1]=g1
    maxVariation=1+epsilon
    innerIter=0
    maxIter=2000
    maxResidual=0
    while((maxVariation>epsilon)and(innerIter<maxIter)):
        innerIter+=1
        maxVariation=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, totalDisplacement, displacement, residuals)
        opt=np.max(residuals)
        if(maxResidual<opt):
            maxResidual=opt
    maxDisplacement=np.max(np.abs(displacement))
    print "Iter: ",innerIter, "Max lateral displacement:", maxDisplacement, "Max variation:",maxVariation, "Max residual:", maxResidual
    return displacement

def estimateMonomodalDeformationField2DMultiScale(leftPyramid, rightPyramid, lambdaParam, level=0, displacementList=None):
    n=len(leftPyramid)
    if(level==(n-1)):
        displacement=estimateNewMonomodalDeformationField2D(leftPyramid[level], rightPyramid[level], lambdaParam, None)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement
    subDisplacement=estimateMonomodalDeformationField2DMultiScale(leftPyramid, rightPyramid, lambdaParam, level+1, displacementList)
    sh=leftPyramid[level].shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]*0.5
    upsampled=np.empty(shape=(leftPyramid[level].shape)+(2,), dtype=np.float64)
    upsampled[:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,0], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,1], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement=estimateNewMonomodalDeformationField2D(leftPyramid[level], rightPyramid[level], lambdaParam, upsampled)
    newDisplacement+=upsampled
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return newDisplacement

def testEstimateMonomodalDeformationField2DMultiScale(lambdaParam):
    fname0='IBSR_01_to_02.nii.gz'
    fname1='data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz'
    nib_left = nib.load(fname0)
    nib_right = nib.load(fname1)
    left=nib_left.get_data().squeeze()
    right=nib_right.get_data().squeeze()
    sl=left.shape
    sr=right.shape
    level=5
    #---sagital---
    left=left[sl[0]//2,:,:].copy()
    right=right[sr[0]//2,:,:].copy()
    #---coronal---
    #left=left[:,sl[1]//2,:].copy()
    #right=right[:,sr[1]//2,:].copy()
    #---axial---
    #left=left[:,:,sl[2]//2].copy()
    #right=right[:,:,sr[2]//2].copy()
    maskLeft=left>0
    maskRight=right>0
    leftPyramid=[img for img in rcommon.pyramid_gaussian_2D(left, level, maskLeft)]
    rightPyramid=[img for img in rcommon.pyramid_gaussian_2D(right, level, maskRight)]
    rcommon.plotOverlaidPyramids(leftPyramid, rightPyramid)
    displacementList=[]
    displacement=estimateMonomodalDeformationField2DMultiScale(leftPyramid, rightPyramid, lambdaParam,0,displacementList)
    warpPyramid=[rcommon.warpImage(leftPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids(warpPyramid, rightPyramid)
    rcommon.overlayImages(warpPyramid[0], rightPyramid[0])
    rcommon.plotDeformationField(displacement)
    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
    maxNorm=np.max(nrm)
    displacement[...,0]*=(maskLeft + maskRight)
    displacement[...,1]*=(maskLeft + maskRight)
    rcommon.plotDeformationField(displacement)
    #nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
    #plt.figure()
    #plt.imshow(nrm)
    print 'Max global displacement: ', maxNorm
    
###############################################################
####### Non-linear Monomodal registration - EM (3D)############
###############################################################

def estimateNewMonomodalDeformationField3D(left, right, lambdaParam, previousDisplacement=None):
    epsilon=1e-3
    sh=left.shape
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
    displacement     =np.zeros(shape=(left.shape)+(3,), dtype=np.float64)
    residuals        =np.zeros(shape=(left.shape), dtype=np.float64)
    gradientField    =np.empty(shape=(left.shape)+(3,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(left.shape)+(3,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
    warped=ndimage.map_coordinates(left, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1], X2+totalDisplacement[...,2]], prefilter=const_prefilter_map_coordinates)
    sigmaField=np.ones_like(warped, dtype=np.float64)
    deltaField=right-warped
    g0, g1, g2=sp.gradient(warped)
    gradientField[:,:,:,0]=g0
    gradientField[:,:,:,1]=g1
    gradientField[:,:,:,2]=g2
    maxVariation=1+epsilon
    innerIter=0
    maxIter=200
    maxResidual=0
    while((maxVariation>epsilon)and(innerIter<maxIter)):
        innerIter+=1
        if(innerIter%100==0):
            print "Iterations:",innerIter, ". Max variation:", maxVariation
        maxVariation=tf.iterateDisplacementField3DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, totalDisplacement, displacement, residuals)
        opt=np.max(residuals)
        if(maxResidual<opt):
            maxResidual=opt
    maxDisplacement=np.max(np.abs(displacement))
    print "Iter: ",innerIter, "Max lateral displacement:", maxDisplacement, "Max variation:",maxVariation, "Max residual:", maxResidual
    return displacement

def estimateMonomodalDeformationField3DMultiScale(leftPyramid, rightPyramid, lambdaParam, level=0, displacementList=None):
    n=len(leftPyramid)
    if(level==(n-1)):
        displacement=estimateNewMonomodalDeformationField3D(leftPyramid[level], rightPyramid[level], lambdaParam, None)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement
    subDisplacement=estimateMonomodalDeformationField3DMultiScale(leftPyramid, rightPyramid, lambdaParam, level+1, displacementList)
    sh=leftPyramid[level].shape
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]*0.5
    upsampled=np.empty(shape=(leftPyramid[level].shape)+(3,), dtype=np.float64)
    upsampled[:,:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,:,0], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,:,1], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,:,2]=ndimage.map_coordinates(subDisplacement[:,:,:,2], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement=estimateNewMonomodalDeformationField3D(leftPyramid[level], rightPyramid[level], lambdaParam, upsampled)
    newDisplacement+=upsampled
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return newDisplacement

def registerNonlinearMonomodal3D(moving, fixed, lambdaParam, levels):
    maskMoving=moving>0
    maskFixed=fixed>0
    movingPyramid=[img for img in rcommon.pyramid_gaussian_3D(moving, levels, maskMoving)]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_3D(fixed, levels, maskFixed)]
    displacement=estimateMonomodalDeformationField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam, 0, None)
    warped=rcommon.warpVolume(movingPyramid[0], displacement)
    return displacement, warped

def testEstimateMonomodalDeformationField3DMultiScale(lambdaParam):
    fname0='IBSR_01_to_02.nii.gz'
    fname1='data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz'
    nib_left = nib.load(fname0)
    nib_right = nib.load(fname1)
    left=nib_left.get_data().squeeze()
    right=nib_right.get_data().squeeze()
    level=5
    maskLeft=left>0
    maskRight=right>0
    leftPyramid=[img for img in rcommon.pyramid_gaussian_3D(left, level, maskLeft)]
    rightPyramid=[img for img in rcommon.pyramid_gaussian_3D(right, level, maskRight)]
    rcommon.plotOverlaidPyramids3DCoronal(leftPyramid, rightPyramid)
    displacementList=[]
    displacement=estimateMonomodalDeformationField3DMultiScale(leftPyramid, rightPyramid, lambdaParam,0,displacementList)
    warpPyramid=[rcommon.warpVolume(leftPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids3DCoronal(warpPyramid, rightPyramid)
    sh=leftPyramid[0].shape
    rcommon.overlayImages(warpPyramid[0][:,sh[1]//4,:], rightPyramid[0][:,sh[1]//4,:])
    rcommon.overlayImages(warpPyramid[0][:,sh[1]//2,:], rightPyramid[0][:,sh[1]//2,:])
    rcommon.overlayImages(warpPyramid[0][:,3*sh[1]//4,:], rightPyramid[0][:,3*sh[1]//4,:])
    rcommon.overlayImages(warpPyramid[0][sh[0]//4,:,:], rightPyramid[0][sh[0]//4,:,:])
    rcommon.overlayImages(warpPyramid[0][sh[0]//2,:,:], rightPyramid[0][sh[0]//2,:,:])
    rcommon.overlayImages(warpPyramid[0][3*sh[0]//4,:,:], rightPyramid[0][3*sh[0]//4,:,:])
    rcommon.overlayImages(warpPyramid[0][:,:,sh[2]//4], rightPyramid[0][:,:,sh[2]//4])
    rcommon.overlayImages(warpPyramid[0][:,:,sh[2]//2], rightPyramid[0][:,:,sh[2]//2])
    rcommon.overlayImages(warpPyramid[0][:,:,3*sh[2]//4], rightPyramid[0][:,:,3*sh[2]//4])
    #rcommon.plotDeformationField(displacement)
    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2 + displacement[...,2]**2)
    maxNorm=np.max(nrm)
    #displacement[...,0]*=(maskLeft + maskRight)
    #displacement[...,1]*=(maskLeft + maskRight)
    #rcommon.plotDeformationField(displacement)
    #figure()
    #imshow(nrm[:,sh[1]//2,:])
    print 'Max global displacement: ', maxNorm
    return displacementList, warpPyramid

###############################################################
####### Non-linear Multimodal registration - EM (2D)###########
###############################################################
def estimateNewMultimodalDeformationField2D(left, right, lambdaParam, quantizationLevels, previousDisplacement=None):
    epsilon=1e-4
    sh=left.shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]
    displacement     =np.empty(shape=(left.shape)+(2,), dtype=np.float64)
    residuals=np.zeros(shape=(left.shape), dtype=np.float64)
    gradientField    =np.empty(shape=(left.shape)+(2,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(left.shape)+(2,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
    rightQ, grayLevels, hist=tf.quantizePositiveImageCYTHON(right, quantizationLevels)
    rightQ=np.array(rightQ, dtype=np.int32)
    finished=False
    maxOuterIter=100
    outerIter=0
    maxDisplacement=None
    maxVariation=None
    maxResidual=0
    while((not finished) and (outerIter<maxOuterIter)):
        outerIter+=1
        print "Outer:", outerIter
        #---E step---
        warped=ndimage.map_coordinates(left, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1]], prefilter=True)
        leftMask=(left>0)*1.0
        warpedLeftMask=ndimage.map_coordinates(leftMask, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1]], order=0, prefilter=False)
        warpedLeftMask=warpedLeftMask.astype(np.int32)
        means, variances=tf.computeMaskedImageClassStatsCYTHON(warpedLeftMask, warped, quantizationLevels, rightQ)
        means[0]=0
        #means, variances=tf.computeImageClassStatsCYTHON(warped, quantizationLevels, rightQ)
        means=np.array(means)
        #variances=np.array([s if s>1e-3 else 1e6 for s in variances])
        #variances[0]=1e-6;
        variances=np.array(variances)
        sigmaField=variances[rightQ]
        deltaField=means[rightQ]-warped
        #--M step--
        g0, g1=sp.gradient(warped)
        gradientField[:,:,0]=g0
        gradientField[:,:,1]=g1
        maxVariation=1+epsilon
        innerIter=0
        maxInnerIter=1000
        displacement[...]=0
        while((maxVariation>epsilon)and(innerIter<maxInnerIter)):
            innerIter+=1
            maxVariation=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, totalDisplacement, displacement, residuals)
            opt=np.max(residuals)
            if(maxResidual<opt):
                maxResidual=opt
        #--accumulate displacement--
        totalDisplacement+=displacement
        #--check stop condition--
        nrm=np.sqrt(displacement[...,0]**2+displacement[...,1]**2)
        maxDisplacement=np.max(nrm)
        if((maxDisplacement<epsilon)or(outerIter>=maxOuterIter)):
            finished=True
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(means[rightQ],cmap=plt.cm.gray)
            plt.title("Estimated warped modality")
            plt.subplot(1,2,2)
            plt.plot(means)
            plt.title("Means")
    print "Iter: ",outerIter, "Max displacement:", maxDisplacement, "Max variation:",maxVariation, "Max residual:", maxResidual
    if(previousDisplacement!=None):
        return totalDisplacement-previousDisplacement
    return totalDisplacement

def estimateMultimodalDeformationField2DMultiScale(leftPyramid, rightPyramid, lambdaParam, level=0, displacementList=None):
    n=len(leftPyramid)
    quantizationLevels=256//(2**level)
    if(level==(n-1)):
        displacement=estimateNewMultimodalDeformationField2D(leftPyramid[level], rightPyramid[level], lambdaParam, quantizationLevels, None)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement
    subDisplacement=estimateMultimodalDeformationField2DMultiScale(leftPyramid, rightPyramid, lambdaParam, level+1, displacementList)
    sh=leftPyramid[level].shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]*0.5
    upsampled=np.empty(shape=(leftPyramid[level].shape)+(2,), dtype=np.float64)
    upsampled[:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,0], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,1], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement=estimateNewMultimodalDeformationField2D(leftPyramid[level], rightPyramid[level], lambdaParam, quantizationLevels, upsampled)
    newDisplacement+=upsampled
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return newDisplacement

def testEstimateMultimodalDeformationField2DMultiScale(lambdaParam):
    fname0='IBSR_01_to_02.nii.gz'
    fname1='data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz'
    nib_left = nib.load(fname0)
    nib_right = nib.load(fname1)
    left=nib_left.get_data().squeeze()
    right=nib_right.get_data().squeeze()
    sl=left.shape
    sr=right.shape
    level=5
    #---sagital---
    #left=left[sl[0]//2,:,:].copy()
    #right=right[sr[0]//2,:,:].copy()
    #---coronal---
    left=left[:,sl[1]//2,:].copy()
    right=right[:,sr[1]//2,:].copy()
    #---axial---
    #left=left[:,:,sl[2]//2].copy()
    #right=right[:,:,sr[2]//2].copy()
    maskLeft=left>0
    maskRight=right>0
    leftPyramid=[img for img in rcommon.pyramid_gaussian_2D(left, level, maskLeft)]
    rightPyramid=[img for img in rcommon.pyramid_gaussian_2D(right, level, maskRight)]
    rcommon.plotOverlaidPyramids(leftPyramid, rightPyramid)
    displacementList=[]
    displacement=estimateMultimodalDeformationField2DMultiScale(leftPyramid, rightPyramid, lambdaParam,0,displacementList)
    warpPyramid=[rcommon.warpImage(leftPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids(warpPyramid, rightPyramid)
    rcommon.overlayImages(warpPyramid[0], rightPyramid[0])
    rcommon.plotDeformationField(displacement)
    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
    maxNorm=np.max(nrm)
    displacement[...,0]*=(maskLeft + maskRight)
    displacement[...,1]*=(maskLeft + maskRight)
    rcommon.plotDeformationField(displacement)
    #nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
    #plt.figure()
    #plt.imshow(nrm)
    print 'Max global displacement: ', maxNorm

###############################################################
####### Non-linear Multimodal registration - EM (3D)###########
###############################################################
def estimateNewMultimodalDeformationField3D(left, right, lambdaParam, maxOuterIter, quantizationLevels, previousDisplacement=None):
    epsilon=1e-3
    sh=left.shape
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
    residuals        =np.zeros(shape=(left.shape),      dtype=np.float64)
    displacement     =np.empty(shape=(left.shape)+(3,), dtype=np.float64)
    gradientField    =np.empty(shape=(left.shape)+(3,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(left.shape)+(3,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
    rightQ, grayLevels, hist=tf.quantizePositiveVolumeCYTHON(right, quantizationLevels)
    rightQ=np.array(rightQ, dtype=np.int32)
    finished=False
    outerIter=0
    maxDisplacement=None
    maxVariation=None
    maxResidual=0
    while((not finished) and (outerIter<maxOuterIter)):
        outerIter+=1
        print "Outer:", outerIter
        #---E step---
        warped=ndimage.map_coordinates(left, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1], X2+totalDisplacement[...,2]], prefilter=True)
        leftMask=(left>0)*1.0
        warpedLeftMask=ndimage.map_coordinates(leftMask, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1], X2+totalDisplacement[...,2]], order=0, prefilter=False)
        warpedLeftMask=warpedLeftMask.astype(np.int32)
        means, variances=tf.computeMaskedVolumeClassStatsCYTHON(warpedLeftMask, warped, quantizationLevels, rightQ)
        means[0]=0
        means=np.array(means)
        variances=np.array(variances)
        sigmaField=variances[rightQ]
        deltaField=means[rightQ]-warped
        #--M step--
        g0, g1, g2=sp.gradient(warped)
        gradientField[:,:,:,0]=g0
        gradientField[:,:,:,1]=g1
        gradientField[:,:,:,2]=g2
        maxVariation=1+epsilon
        innerIter=0
        maxInnerIter=100
        displacement[...]=0
        while((maxVariation>epsilon)and(innerIter<maxInnerIter)):
            innerIter+=1
            maxVariation=tf.iterateDisplacementField3DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, totalDisplacement, displacement, residuals)
            opt=np.max(residuals)
            if(maxResidual<opt):
                maxResidual=opt
        #--accumulate displacement--
        totalDisplacement+=displacement
        #--check stop condition--
        nrm=np.sqrt(displacement[...,0]**2+displacement[...,1]**2+displacement[...,2]**2)
        maxDisplacement=np.max(nrm)
        if((maxDisplacement<epsilon)or(outerIter>=maxOuterIter)):
            finished=True
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(means[rightQ[:,sh[1]//2,:]],cmap=plt.cm.gray)
            plt.title("Estimated warped modality")
            plt.subplot(1,2,2)
            plt.plot(means)
            plt.title("Means")
    print "Iter: ",outerIter, "Max displacement:", maxDisplacement, "Max variation:",maxVariation, "Max residual:", maxResidual
    if(previousDisplacement!=None):
        return totalDisplacement-previousDisplacement
    return totalDisplacement

def estimateMultimodalDeformationField3DMultiScale(leftPyramid, rightPyramid, lambdaParam, maxOuterIter, level=0, displacementList=None):
    n=len(leftPyramid)
    quantizationLevels=256//(2**level)
    if(level==(n-1)):
        displacement=estimateNewMultimodalDeformationField3D(leftPyramid[level], rightPyramid[level], lambdaParam, maxOuterIter[level], quantizationLevels, None)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement
    subDisplacement=estimateMultimodalDeformationField3DMultiScale(leftPyramid, rightPyramid, lambdaParam, maxOuterIter, level+1, displacementList)
    sh=leftPyramid[level].shape
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]*0.5
    upsampled=np.empty(shape=(leftPyramid[level].shape)+(3,), dtype=np.float64)
    upsampled[:,:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,:,0], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,:,1], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,:,2]=ndimage.map_coordinates(subDisplacement[:,:,:,2], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement=estimateNewMultimodalDeformationField3D(leftPyramid[level], rightPyramid[level], lambdaParam, maxOuterIter[level], quantizationLevels, upsampled)
    newDisplacement+=upsampled
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return newDisplacement
    
    
def registerNonlinearMultimodal3D(moving, fixed, lambdaParam, levels):
    maskMoving=moving>0
    maskFixed=fixed>0
    movingPyramid=[img for img in rcommon.pyramid_gaussian_3D(moving, levels, maskMoving)]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_3D(fixed, levels, maskFixed)]
    maxOuterIter=[10,50,100,100,100,100]
    displacement=estimateMultimodalDeformationField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0, None)
    warped=rcommon.warpVolume(movingPyramid[0], displacement)
    return displacement, warped


def testEstimateMultimodalDeformationField3DMultiScale(lambdaParam):
    fname0='IBSR_01_to_02.nii.gz'
    fname1='data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz'
    nib_left = nib.load(fname0)
    nib_right = nib.load(fname1)
    left=nib_left.get_data().squeeze()
    right=nib_right.get_data().squeeze()
    level=5
    maskLeft=left>0
    maskRight=right>0
    leftPyramid=[img for img in rcommon.pyramid_gaussian_3D(left, level, maskLeft)]
    rightPyramid=[img for img in rcommon.pyramid_gaussian_3D(right, level, maskRight)]
    rcommon.plotOverlaidPyramids3DCoronal(leftPyramid, rightPyramid)
    maxOuterIter=[10,50,100,100,100,100]
    displacementList=[]
    displacement=estimateMultimodalDeformationField3DMultiScale(leftPyramid, rightPyramid, lambdaParam, maxOuterIter,0,displacementList)
    warpPyramid=[rcommon.warpVolume(leftPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids3DCoronal(warpPyramid, rightPyramid)
    sh=leftPyramid[0].shape
    rcommon.overlayImages(warpPyramid[0][:,sh[1]//4,:], rightPyramid[0][:,sh[1]//4,:])
    rcommon.overlayImages(warpPyramid[0][:,sh[1]//2,:], rightPyramid[0][:,sh[1]//2,:])
    rcommon.overlayImages(warpPyramid[0][:,3*sh[1]//4,:], rightPyramid[0][:,3*sh[1]//4,:])
    rcommon.overlayImages(warpPyramid[0][sh[0]//4,:,:], rightPyramid[0][sh[0]//4,:,:])
    rcommon.overlayImages(warpPyramid[0][sh[0]//2,:,:], rightPyramid[0][sh[0]//2,:,:])
    rcommon.overlayImages(warpPyramid[0][3*sh[0]//4,:,:], rightPyramid[0][3*sh[0]//4,:,:])
    rcommon.overlayImages(warpPyramid[0][:,:,sh[2]//4], rightPyramid[0][:,:,sh[2]//4])
    rcommon.overlayImages(warpPyramid[0][:,:,sh[2]//2], rightPyramid[0][:,:,sh[2]//2])
    rcommon.overlayImages(warpPyramid[0][:,:,3*sh[2]//4], rightPyramid[0][:,:,3*sh[2]//4])
    #rcommon.plotDeformationField(displacement)
    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2 + displacement[...,2]**2)
    maxNorm=np.max(nrm)
    #displacement[...,0]*=(maskLeft + maskRight)
    #displacement[...,1]*=(maskLeft + maskRight)
    #rcommon.plotDeformationField(displacement)
    #figure()
    #imshow(nrm[:,sh[1]//2,:])
    print 'Max global displacement: ', maxNorm
    return displacementList, warpPyramid    
    
if __name__=="__main__":
    #---Monomodal 3D---
    lambdaParam=150
    displacementList, warpPyramid=testEstimateMonomodalDeformationField3DMultiScale(lambdaParam)
    #---Multimodal 3D---
    t1_template=nib.load('IBSR_template_to_01.nii.gz')
    t1_template=t1_template.get_data().squeeze()
    t1_real=nib.load('data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz')
    t1_real=t1_real.get_data().squeeze()
    t2_template=nib.load('IBSR_t2template_to_01.nii.gz')
    t2_template=t2_template.get_data().squeeze()
    #This is the "ground truth": register t1_template to t1_real    
    displacement, warped=registerNonlinearMonomodal3D(t1_template, t1_real, lambdaParam, 3)
    #now try to reproduce using multimodal: t2_template to t1_real
    displacement, warped=registerNonlinearMultimodal3D(t2_template, t1_real, lambdaParam, 3)
    #now synthetically deform template[T1] and try to recover the deformation field using
    #template T2
    t1displacement=np.load('template_to_01.npy')
    t1templateWarped=rcommon.warpVolume(t1_template, t1displacement)
    dispRecovered, warpedRecovered=registerNonlinearMultimodal3D(t2_template, t1templateWarped, lambdaParam, 3)
    
    
    
    t2_template=nib.load('data/t2/t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')
    t2_template=t2_template.get_data().squeeze()
    rcommon.overlayImages(t2_template[:,64,:],t1templateWarped[:,64,:])
    
    left=np.fromfile('data/t2/t2_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape(ns,nr,nc)
    left=left.astype(np.float64)
    right=np.fromfile('data/t1/t1_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape(ns,nr,nc)
    right=right.astype(np.float64)
    
    
#    The objective is to register these two volumes:
#    np.save('t2_template.npy',t2_template)
#    np.save('t1_templateWarped.npy',t1templateWarped)
#    the ground truth is 'template_to_01.npy', which was used to deform the T1 
#    template. This was my first attempt:
#     dispRecovered, warpedRecovered=registerNonlinearMultimodal3D(t2_template, t1templateWarped, lambdaParam, 3)