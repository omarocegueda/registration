# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:47:00 2013

@author: khayyam
"""
import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import misc
from skimage import transform
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import registrationCommon as rcommon
import tensorFieldUtils as tf
from nipy.core.api import Image, AffineTransform
import nibabel as nib
import nipy
from registrationCommon import const_prefilter_map_coordinates
###############################################################
##########################Rotation only########################
###############################################################

def estimateRotationOnly(left, right, maxAngleRads=0, thr=-1):
    center=(np.array(right.shape)-1)/2.0
    C,R=sp.meshgrid(np.array(range(right.shape[1]), dtype=np.float64), np.array(range(right.shape[0]), dtype=np.float64))
    R=R-center[0]
    C=C-center[1]
    [dr, dc]=sp.gradient(right)
    prod=R*dc-C*dr
    diff=left-right
    num=prod*diff
    den=prod**2
    if(thr>0):
        N=np.sqrt((R*maxAngleRads)**2+(C*maxAngleRads)**2)
        M=(N<thr)
        theta=np.sum(num*M)/np.sum(den*M)
    else:
        theta=np.sum(num)/np.sum(den)
    return theta

def estimateNewRotation(left, right, previousAngleRadians, maxAngleRads=0, thr=-1):
    epsilon=1e-9
    center=(np.array(right.shape)-1)/2.0
    C,R=sp.meshgrid(np.array(range(right.shape[1]), dtype=np.float64), np.array(range(right.shape[0]), dtype=np.float64))
    R=R-center[0]
    C=C-center[1]
    if(np.abs(previousAngleRadians)>epsilon):
        a=np.cos(previousAngleRadians)
        b=np.sin(previousAngleRadians)
        Rnew,Cnew=(a*R-b*C+center[0], b*R+a*C+center[1])
        right=ndimage.map_coordinates(right, [Rnew,Cnew], prefilter=const_prefilter_map_coordinates)
    [dr, dc]=sp.gradient(right)
    prod=R*dc-C*dr
    diff=left-right
    num=prod*diff
    den=prod**2
    theta=0
    if(thr>0):
        N=np.sqrt((R*maxAngleRads)**2+(C*maxAngleRads)**2)
        M=(N<thr)
        theta=np.sum(num*M)/np.sum(den*M)
    else:
        theta=np.sum(num)/np.sum(den)
    return theta

def estimateRotationMultiscale(leftPyramid, rightPyramid, level=0, rotationList=None):
    n=len(leftPyramid)
    if(level>=n):
        return 0
    if(level==(n-1)):
        theta=estimateRotationOnly(leftPyramid[level], rightPyramid[level])
        if(rotationList!=None):
            rotationList.append(theta)
        return theta
    thetaSub=estimateRotationMultiscale(leftPyramid, rightPyramid, level+1, rotationList)
    theta=estimateNewRotation(leftPyramid[level], rightPyramid[level], thetaSub)
    if(rotationList!=None):
        rotationList.append(theta)
    return theta+thetaSub    

def testEstimateRotationMultiscale(dTheta, level):
    #inImg=misc.imread('stinkbug.png')[...,0]
    inImg=misc.imread('T2sample.png')[...,0]
    left=ndimage.rotate(inImg, -dTheta/2.0)
    right=ndimage.rotate(inImg, dTheta/2.0)
    rightPyramid=[i for i in transform.pyramid_gaussian(right, level)]
    leftPyramid=[i for i in transform.pyramid_gaussian(left, level)]
    angles=[]
    theta=estimateRotationMultiscale(leftPyramid, rightPyramid, 0, angles)
    angles=180*np.array(angles).reshape(len(angles),1)/np.pi
    xticks=[str(i) for i in range(level+1)[::-1]]
    plt.figure()
    plt.plot(angles)
    plt.xlabel('Scale')
    plt.ylabel('Estimated angle')
    plt.title('Global rotation [GT/Est.]: '+str(dTheta)+'/'+str(180*theta/np.pi)+' deg. Levels: '+str(level+1))
    plt.xticks(range(level+1), xticks)
    plt.grid()
    #printPyramids(leftPyramid, rightPyramid)
    print 'Estimated:\n', angles,' degrees'
    return 180*theta/np.pi

###############################################################
######################Rigid transformation#####################
###############################################################

def estimateRigidTransformation(left, right):
    #compute the centered meshgrid
    center=(np.array(right.shape)-1)/2.0
    C,R=sp.meshgrid(np.array(range(right.shape[1]), dtype=np.float64), np.array(range(right.shape[0]), dtype=np.float64))
    R=R-center[0]
    C=C-center[1]
    #parameter estimation
    [dr, dc]=sp.gradient(right)
    epsilon=R*dc-C*dr
    c=np.array([epsilon,dr,dc]).transpose(1,2,0)
    tensorProds=c[:,:,:,None]*c[:,:,None,:]
    A=np.sum(np.sum(tensorProds, axis=0), axis=0)
    diff=left-right
    prod=c*diff[:,:,None]
    b=np.sum(np.sum(prod, axis=0), axis=0)
    beta=linalg.solve(A,b)
    return beta

def estimateNewRigidTransformation(left, right, previousBeta=None):
    epsilon=1e-9
    center=(np.array(right.shape)-1)/2.0
    C,R=sp.meshgrid(np.array(range(right.shape[1]), dtype=np.float64), np.array(range(right.shape[0]), dtype=np.float64))
    R=R-center[0]
    C=C-center[1]    
    if((previousBeta!=None) and (np.max(np.abs(previousBeta))>epsilon)):
        a=np.cos(previousBeta[0])
        b=np.sin(previousBeta[0])
        Rnew,Cnew=(a*R-b*C+2.0*previousBeta[1]+center[0], b*R+a*C+2.0*previousBeta[2]+center[1])
        right=ndimage.map_coordinates(right, [Rnew,Cnew], prefilter=const_prefilter_map_coordinates)
    [dr, dc]=sp.gradient(right)
    epsilon=R*dc-C*dr
    c=np.array([epsilon,dr,dc]).transpose(1,2,0)
    tensorProds=c[:,:,:,None]*c[:,:,None,:]
    A=np.sum(np.sum(tensorProds, axis=0), axis=0)
    diff=left-right
    prod=c*diff[:,:,None]
    b=np.sum(np.sum(prod, axis=0), axis=0)
    beta=linalg.solve(A,b)
    return beta

def estimateRigidTransformationMultiscale(leftPyramid, rightPyramid, level=0, paramList=None):
    n=len(leftPyramid)
    if(level>=n):
        return np.array([0,0,0])
    if(level==(n-1)):
        beta=estimateNewRigidTransformation(leftPyramid[level], rightPyramid[level])
        if(paramList!=None):
            paramList.append(beta)
        return beta
    betaSub=estimateRigidTransformationMultiscale(leftPyramid, rightPyramid, level+1, paramList)
    beta=estimateNewRigidTransformation(leftPyramid[level], rightPyramid[level], betaSub)
    if(paramList!=None):
        paramList.append(beta)
    return beta+(betaSub*np.array([1.0,2.0,2.0]))

def testRigidTransformationMultiscale(dTheta, displacement, level):
    inImg=misc.imread('T2sample.png')[...,0]
    left=ndimage.rotate(inImg, -0.5*dTheta)#Rotate symmetricaly to ensure both are still the same size
    right=ndimage.rotate(inImg, 0.5*dTheta)#Rotate symmetricaly to ensure both are still the same size
    right=ndimage.affine_transform(right, np.eye(2), offset=-1*displacement)
    rightPyramid=[i for i in transform.pyramid_gaussian(right, level)]
    leftPyramid=[i for i in transform.pyramid_gaussian(left, level)]
    rcommon.plotPyramids(leftPyramid, rightPyramid)
    beta=estimateRigidTransformationMultiscale(leftPyramid, rightPyramid)
    print 180.0*beta[0]/np.pi, beta[1:3]
    return beta

def testRigidTransformEstimation(inImg, level, dTheta, displacement, thr):
    left=ndimage.rotate(inImg, dTheta)
    right=ndimage.rotate(inImg, -dTheta)
    left=ndimage.affine_transform(left , np.eye(2), offset=-1*displacement)
    right=ndimage.affine_transform(right, np.eye(2), offset=displacement)
    
    rightPyramid=[i for i in transform.pyramid_gaussian(right, level)]
    leftPyramid=[i for i in transform.pyramid_gaussian(left, level)]
    sel=level
    beta=estimateRigidTransformation(leftPyramid[sel], rightPyramid[sel], 2.0*dTheta, thr)
    return beta

#def runRotationExperiment(inImage, dTheta):
#    plotTrustRegions(inImage, 3, dTheta, [0.5,1,1.5,2])
#    T=np.array([testRotationEstimation(inImage, 3, dTheta, thr) for thr in np.linspace(0.2,3, 100)])
#    D=np.abs(T-(-2.0*dTheta*np.pi/180))*180.0/np.pi
#    plt.figure()
#    plt.plot(np.linspace(0.2,3, 100),D)
#    plt.xlabel('Norm threshold')
#    plt.ylabel('Error (degrees)')
#    plt.title('Estimation error. Rotation: '+str(2*dTheta)+' degrees.')

def runRigidTransformExperiment(inImage, dTheta, displacement):
    beta=testRigidTransformEstimation(inImage, 3, dTheta, displacement, 1.0)
    print beta

###############################################################
######################Rigid transformation#####################
###############################################################

def estimateNewRigidTransformation3D(left, right, previousBeta=None):
    epsilon=1e-9
    sh=left.shape
    center=(np.array(sh)-1)/2.0
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
    X0=X0-center[0]
    X1=X1-center[1]
    X2=X2-center[2]
    if((previousBeta!=None) and (np.max(np.abs(previousBeta))>epsilon)):
        R=rcommon.getRotationMatrix(previousBeta[0:3])
        X0new,X1new,X2new=(R[0,0]*X0 + R[0,1]*X1 + R[0,2]*X2 + center[0] + 2.0*previousBeta[3], 
                        R[1,0]*X0 + R[1,1]*X1 + R[1,2]*X2 + center[1] + 2.0*previousBeta[4], 
                        R[2,0]*X0 + R[2,1]*X1 + R[2,2]*X2 + center[2] + 2.0*previousBeta[5])
        left=ndimage.map_coordinates(left, [X0new, X1new, X2new], prefilter=const_prefilter_map_coordinates)
    g0, g1, g2=sp.gradient(left)
    q=np.empty(shape=(X0.shape)+(6,), dtype=np.float64)
    q[...,0]=g2*X1-g1*X2
    q[...,1]=g0*X2-g2*X0
    q[...,2]=g1*X0-g0*X1
    q[...,3]=g0
    q[...,4]=g1
    q[...,5]=g2
    diff=right-left
    A,b=tf.integrateTensorFieldProductsCYTHON(q,diff)
    #A,b=tf.integrateTensorFieldProducts(q,diff)
    beta=linalg.solve(A,b)
    return beta

def estimateRigidTransformationMultiscale3D(leftPyramid, rightPyramid, level=0, paramList=None):
    n=len(leftPyramid)
    if(level>=n):
        return np.array([0,0,0,0,0,0], dtype=np.float64)
    if(level==(n-1)):
        solution=None
        for i in range(level+1):
            beta=estimateNewRigidTransformation3D(leftPyramid[level], rightPyramid[level], solution)
            solution=beta if solution==None else solution+beta
        if(paramList!=None):
            paramList.append(beta)
        print 'Level',level,': ',180*beta[:3]/np.pi, beta[3:]
        return beta
    betaSub=estimateRigidTransformationMultiscale3D(leftPyramid, rightPyramid, level+1, paramList)
    for i in range(2*(level+1)):
        beta=estimateNewRigidTransformation3D(leftPyramid[level], rightPyramid[level], betaSub)
        betaSub+=beta*np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    if(paramList!=None):
        paramList.append(beta)
    beta=betaSub*np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    print 'Level ',level,': ',180*beta[:3]/np.pi, beta[3:]
    return beta

def plotSlicePyramidsAxial(L, R):
    n=len(L)
    plt.figure()
    for i in range(n):
        nSlices=L[i].shape[0]
        plt.subplot(2,n,i+1)
        plt.imshow(L[i][nSlices//2,...], cmap=plt.cm.gray)
        plt.title('Level: '+str(i))
        plt.subplot(2,n,n+i+1)
        plt.imshow(R[i][nSlices//2,...], cmap=plt.cm.gray)
        plt.title('Level: '+str(i))

def plotSlicePyramidsCoronal(L, R):
    n=len(L)
    plt.figure()
    for i in range(n):
        nRows=L[i].shape[1]
        plt.subplot(2,n,i+1)
        plt.imshow(L[i][:,nRows//2,...], cmap=plt.cm.gray)
        plt.title('Level: '+str(i))
        plt.subplot(2,n,n+i+1)
        plt.imshow(R[i][:,nRows//2,...], cmap=plt.cm.gray)
        plt.title('Level: '+str(i))

def testRigidTransformationMultiScale3D(betaGT, level):
    betaGTRads=np.array(betaGT, dtype=np.float64)
    betaGTRads[0:3]=np.copy(np.pi*betaGTRads[0:3]/180.0)
    ns=181
    nr=217
    nc=181
    print 'Loading volume...'
    inImg=np.fromfile('data/t2/t2_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape(ns,nr,nc)
    inImg=inImg.astype(np.float64)
    print 'Generating pyramid at level',level,'...'
    left=inImg
    right=rcommon.applyRigidTransformation3D(inImg, betaGTRads)
    leftPyramid=[i for i in rcommon.pyramid_gaussian_3D(left, level)]
    rightPyramid=[i for i in rcommon.pyramid_gaussian_3D(right, level)]
    plotSlicePyramidsAxial(leftPyramid, rightPyramid)
    print 'Estimation started.'
    beta=estimateRigidTransformationMultiscale3D(leftPyramid, rightPyramid)
    print 'Estimation finished.'
    print 'Ground truth:', betaGT
    return beta

###############################################################
################## Rigid - Multimodal - EM ####################
###############################################################
def plotEstimatedTransferedImage(v, numLabels, labels, beta):
    vtrans=rcommon.applyRigidTransformation3D(v,beta)
    means, variances=tf.computeVolumeClassStatsCYTHON(vtrans, numLabels, labels)
    means=np.array(means)
    variances=np.array(variances)
    sh=labels.shape
    meansSlice=means[labels[sh[0]//2]]
    varianceSlice=variances[labels[sh[0]//2]]
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('Means ['+str(numLabels)+']')
    plt.imshow(meansSlice, cmap=plt.cm.gray)
    plt.subplot(1,2,2)
    plt.imshow(varianceSlice, cmap=plt.cm.gray)
    plt.title('Variances ['+str(numLabels)+']')

def plotToneTransferFunction(left, rightQ, numLevels, beta):
    transformed=rcommon.applyRigidTransformation3D(left, beta)
    data=[transformed[rightQ==i] for i in range(numLevels)]
    plt.figure()
    plt.boxplot(data, 0, '')

def estimateNewMultimodalRigidTransformation3D(left, right, rightQ, numLevels, previousBeta=None):
    epsilon=1e-9
    sh=left.shape
    center=(np.array(sh)-1)/2.0
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
    X0=X0-center[0]
    X1=X1-center[1]
    X2=X2-center[2]
    mask=np.ones_like(X0, dtype=np.int32)
    if((previousBeta!=None) and (np.max(np.abs(previousBeta))>epsilon)):
        R=rcommon.getRotationMatrix(previousBeta[0:3])
        X0new,X1new,X2new=(R[0,0]*X0 + R[0,1]*X1 + R[0,2]*X2 + center[0] + 2.0*previousBeta[3], 
                        R[1,0]*X0 + R[1,1]*X1 + R[1,2]*X2 + center[1] + 2.0*previousBeta[4], 
                        R[2,0]*X0 + R[2,1]*X1 + R[2,2]*X2 + center[2] + 2.0*previousBeta[5])
        left=ndimage.map_coordinates(left, [X0new, X1new, X2new], prefilter=const_prefilter_map_coordinates)
        mask[...]=(X0new<0) + (X0new>(sh[0]-1))
        mask[...]=mask + (X1new<0) + (X1new>(sh[1]-1))
        mask[...]=mask + (X2new<0) + (X2new>(sh[2]-1))
        mask[...]=1-mask
    means, variances=tf.computeMaskedVolumeClassStatsCYTHON(mask, left, numLevels, rightQ)
    means=np.array(means)
    weights=np.array([1.0/x if(x>0) else 0 for x in variances], dtype=np.float64)
    g0, g1, g2=sp.gradient(left)
    q=np.empty(shape=(X0.shape)+(6,), dtype=np.float64)
    q[...,0]=g2*X1-g1*X2
    q[...,1]=g0*X2-g2*X0
    q[...,2]=g1*X0-g0*X1
    q[...,3]=g0
    q[...,4]=g1
    q[...,5]=g2
    diff=means[rightQ]-left
    Aw, bw=tf.integrateMaskedWeightedTensorFieldProductsCYTHON(mask, q, diff, numLevels, rightQ, weights)
    beta=linalg.solve(Aw,bw)
    return beta

def estimateMultiModalRigidTransformationMultiscale3D(leftPyramid, rightPyramid, level=0, paramList=None):
    n=len(leftPyramid)
    quantizationLevels=256//(2**level)
    if(level>=n):
        return np.array([0,0,0,0,0,0], dtype=np.float64)
    if(level==(n-1)):
        solution=None
        rightQ, grayLevels, hist=tf.quantizeVolumeCYTHON(rightPyramid[level], quantizationLevels)
        rightQ=np.array(rightQ, dtype=np.int32)
        for i in range(level+1):
            beta=estimateNewMultimodalRigidTransformation3D(leftPyramid[level], rightPyramid[level], rightQ, quantizationLevels,solution)
            solution=beta if solution==None else solution+beta
        if(paramList!=None):
            paramList.append(beta)
        print 'Level',level,': ',180*beta[:3]/np.pi, beta[3:]
        #plotToneTransferFunction(leftPyramid[level], rightQ, quantizationLevels, beta)
        return beta
    betaSub=estimateMultiModalRigidTransformationMultiscale3D(leftPyramid, rightPyramid, level+1, paramList)
    rightQ, grayLevels, hist=tf.quantizeVolumeCYTHON(rightPyramid[level], quantizationLevels)
    rightQ=np.array(rightQ, dtype=np.int32)
    for i in range(2*(level+1)):
        beta=estimateNewMultimodalRigidTransformation3D(leftPyramid[level], rightPyramid[level], rightQ, quantizationLevels, betaSub)
        betaSub+=beta*np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
    if(paramList!=None):
        paramList.append(beta)
    beta=betaSub*np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    print 'Level ',level,': ',180*beta[:3]/np.pi, beta[3:]
    plotEstimatedTransferedImage(leftPyramid[level], quantizationLevels, rightQ, beta)
    #plotToneTransferFunction(leftPyramid[level], rightQ, quantizationLevels, beta)
    return beta

def testMultimodalRigidTransformationMultiScale3D(betaGT, level):
    betaGTRads=np.array(betaGT, dtype=np.float64)
    betaGTRads[0:3]=np.copy(np.pi*betaGTRads[0:3]/180.0)
    ns=181
    nr=217
    nc=181
    print 'Loading volume...'
    left=np.fromfile('data/t2/t2_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape(ns,nr,nc)
    left=left.astype(np.float64)
    print 'Generating pyramid at level',level,'...'
    right=np.fromfile('data/t1/t1_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape(ns,nr,nc)
    right=right.astype(np.float64)
    right=rcommon.applyRigidTransformation3D(right, betaGTRads)
    leftPyramid=[i for i in rcommon.pyramid_gaussian_3D(left, level)]
    rightPyramid=[i for i in rcommon.pyramid_gaussian_3D(right, level)]
    print 'Estimation started.'
    beta=estimateMultiModalRigidTransformationMultiscale3D(leftPyramid, rightPyramid)
    print 'Estimation finished.'
    print 'Ground truth:', betaGT
    plotSlicePyramidsAxial(leftPyramid, rightPyramid)
    return beta


def generateTestingPair(betaGT):
    betaGTRads=np.array(betaGT, dtype=np.float64)
    betaGTRads[0:3]=np.copy(np.pi*betaGTRads[0:3]/180.0)
    ns=181
    nr=217
    nc=181
    left=np.fromfile('data/t2/t2_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape(ns,nr,nc)
    left=left.astype(np.float64)
    right=np.fromfile('data/t1/t1_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape(ns,nr,nc)
    right=right.astype(np.float64)
    right=rcommon.applyRigidTransformation3D(right, betaGTRads)
    affine_transform=AffineTransform('ijk', ['aligned-z=I->S','aligned-y=P->A', 'aligned-x=L->R'], np.eye(4))
    left=Image(left, affine_transform)
    right=Image(right, affine_transform)
    nipy.save_image(left,'moving.nii')
    nipy.save_image(right,'fixed.nii')

def testIntersubjectRigidRegistration(fname0, fname1, level, outfname):
    nib_left = nib.load(fname0)
    nib_right = nib.load(fname1)
    left=nib_left.get_data().astype(np.double).squeeze()
    right=nib_right.get_data().astype(np.double).squeeze()
    leftPyramid=[i for i in rcommon.pyramid_gaussian_3D(left, level)]
    rightPyramid=[i for i in rcommon.pyramid_gaussian_3D(right, level)]
    plotSlicePyramidsAxial(leftPyramid, rightPyramid)
    print 'Estimation started.'
    beta=estimateRigidTransformationMultiscale3D(leftPyramid, rightPyramid)
    print 'Estimation finished.'
    rcommon.applyRigidTransformation3D(left, beta)
    sl=np.array(left.shape)//2
    sr=np.array(right.shape)//2
    rcommon.overlayImages(left[sl[0],:,:], right[sr[0],:,:])
    affine_transform=AffineTransform('ijk', ['aligned-z=I->S','aligned-y=P->A', 'aligned-x=L->R'], np.eye(4))
    left=Image(left, affine_transform)
    nipy.save_image(left,outfname)
    return beta

def peelTemplateBrain():
    ns=181
    nr=217
    nc=181
    gt_template=np.fromfile('data/phantom_1.0mm_normal_crisp.rawb', dtype=np.ubyte).reshape((ns,nr,nc))
    t1_template=np.fromfile('data/t1/t1_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape((ns,nr,nc))
    t2_template=np.fromfile('data/t2/t2_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape((ns,nr,nc))
    #t1_template*=((1<=gt_template)*(gt_template<=3)+(gt_template==8))
    t1_template*=((1<=gt_template)*(gt_template<=3))
    t2_template*=((1<=gt_template)*(gt_template<=3))
    affine_transform=AffineTransform('ijk', ['aligned-z=I->S','aligned-y=P->A', 'aligned-x=L->R'], np.eye(4))
    t1_template=Image(t1_template, affine_transform)
    t2_template=Image(t2_template, affine_transform)
    nipy.save_image(t1_template,'data/t1/t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')
    nipy.save_image(t2_template,'data/t2/t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')

def generateRegistrationScript():
    with open('register_all.sh','w') as f:
        for i in range(18):
            idxi='0'+str(i+1) if(i<9) else str(i+1)
            for j in range(18):
                if(i!=j):
                    idxj='0'+str(j+1) if(j<9) else str(j+1)
                    referenceFile='data/t1/IBSR18/IBSR_'+idxj+'/IBSR_'+idxj+'_ana_strip.nii.gz'
                    inputFile    ='data/t1/IBSR18/IBSR_'+idxi+'/IBSR_'+idxi+'_ana_strip.nii.gz'
                    outFile      ='IBSR_'+idxi+'_to_'+idxj+'.nii'
                    outMatrix    ='IBSR_'+idxi+'_to_'+idxj+'.txt'
                    f.write('flirt -in '+inputFile+' -ref '+referenceFile+' -omat '+outMatrix+' -out '+outFile+'\n')
