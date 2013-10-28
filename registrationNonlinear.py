import numpy as np
import scipy as sp
import tensorFieldUtils as tf
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
import registrationCommon as rcommon
from registrationCommon import const_prefilter_map_coordinates
import ecqmmf
import os.path

###############################################################
####### Non-linear Monomodal registration - EM (2D)############
###############################################################

def estimateNewMonomodalDeformationField2D(moving, fixed, lambdaParam, maxIter, previousDisplacement=None):
    epsilon=1e-5
    sh=moving.shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]
    displacement     =np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
    residuals=np.zeros(shape=(moving.shape), dtype=np.float64)
    gradientField    =np.empty(shape=(moving.shape)+(2,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
    warped=ndimage.map_coordinates(moving, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1]], prefilter=const_prefilter_map_coordinates)
    sigmaField=np.ones_like(warped, dtype=np.float64)
    deltaField=fixed-warped
    g0, g1=sp.gradient(warped)
    gradientField[:,:,0]=g0
    gradientField[:,:,1]=g1
    maxVariation=1+epsilon
    innerIter=0
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

def estimateMonomodalDeformationField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, level=0, displacementList=None):
    n=len(movingPyramid)
    if(level==(n-1)):
        displacement=estimateNewMonomodalDeformationField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, None)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement
    subDisplacement=estimateMonomodalDeformationField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, level+1, displacementList)
    sh=movingPyramid[level].shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]*0.5
    upsampled=np.empty(shape=(movingPyramid[level].shape)+(2,), dtype=np.float64)
    upsampled[:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,0], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,1], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement=estimateNewMonomodalDeformationField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, upsampled)
    newDisplacement+=upsampled
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return newDisplacement

def testEstimateMonomodalDeformationField2DMultiScale(lambdaParam):
    fname0='IBSR_01_to_02.nii.gz'
    fname1='data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz'
    nib_moving = nib.load(fname0)
    nib_fixed= nib.load(fname1)
    moving=nib_moving.get_data().squeeze()
    fixed=nib_fixed.get_data().squeeze()
    sl=moving.shape
    sr=fixed.shape
    level=5
    #---sagital---
    moving=moving[sl[0]//2,:,:].copy()
    fixed=fixed[sr[0]//2,:,:].copy()
    #---coronal---
    #moving=moving[:,sl[1]//2,:].copy()
    #fixed=fixed[:,sr[1]//2,:].copy()
    #---axial---
    #moving=moving[:,:,sl[2]//2].copy()
    #fixed=fixed[:,:,sr[2]//2].copy()
    maskMoving=moving>0
    maskFixed=fixed>0
    movingPyramid=[img for img in rcommon.pyramid_gaussian_2D(moving, level, maskMoving)]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_2D(fixed, level, maskFixed)]
    rcommon.plotOverlaidPyramids(movingPyramid, fixedPyramid)
    displacementList=[]
    displacement=estimateMonomodalDeformationField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam,0,displacementList)
    warpPyramid=[rcommon.warpImage(movingPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids(warpPyramid, fixedPyramid)
    rcommon.overlayImages(warpPyramid[0], fixedPyramid[0])
    rcommon.plotDeformationField(displacement)
    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
    maxNorm=np.max(nrm)
    displacement[...,0]*=(maskMoving + maskFixed)
    displacement[...,1]*=(maskMoving + maskFixed)
    rcommon.plotDeformationField(displacement)
    #nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
    #plt.figure()
    #plt.imshow(nrm)
    print 'Max global displacement: ', maxNorm

###############################################################
####### Non-linear Monomodal registration - EM (3D)############
###############################################################

def estimateNewMonomodalDeformationField3D(moving, fixed, lambdaParam, maxIter, previousDisplacement=None):
    epsilon=1e-3
    sh=moving.shape
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
    displacement     =np.zeros(shape=(moving.shape)+(3,), dtype=np.float64)
    residuals        =np.zeros(shape=(moving.shape), dtype=np.float64)
    gradientField    =np.empty(shape=(moving.shape)+(3,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(moving.shape)+(3,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
    warped=ndimage.map_coordinates(moving, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1], X2+totalDisplacement[...,2]], prefilter=const_prefilter_map_coordinates)
    sigmaField=np.ones_like(warped, dtype=np.float64)
    deltaField=fixed-warped
    g0, g1, g2=sp.gradient(warped)
    gradientField[:,:,:,0]=g0
    gradientField[:,:,:,1]=g1
    gradientField[:,:,:,2]=g2
    maxVariation=1+epsilon
    innerIter=0
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

def estimateMonomodalDeformationField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam, level=0, displacementList=None):
    n=len(movingPyramid)
    if(level==(n-1)):
        displacement=estimateNewMonomodalDeformationField3D(movingPyramid[level], fixedPyramid[level], lambdaParam, None)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement
    subDisplacement=estimateMonomodalDeformationField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam, level+1, displacementList)
    sh=movingPyramid[level].shape
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]*0.5
    upsampled=np.empty(shape=(movingPyramid[level].shape)+(3,), dtype=np.float64)
    upsampled[:,:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,:,0], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,:,1], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,:,2]=ndimage.map_coordinates(subDisplacement[:,:,:,2], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement=estimateNewMonomodalDeformationField3D(movingPyramid[level], fixedPyramid[level], lambdaParam, upsampled)
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
    nib_moving = nib.load(fname0)
    nib_fixed = nib.load(fname1)
    moving=nib_moving.get_data().squeeze()
    fixed=nib_fixed.get_data().squeeze()
    level=5
    maskMoving=moving>0
    maskFixed=fixed>0
    movingPyramid=[img for img in rcommon.pyramid_gaussian_3D(moving, level, maskMoving)]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_3D(fixed, level, maskFixed)]
    rcommon.plotOverlaidPyramids3DCoronal(movingPyramid, fixedPyramid)
    displacementList=[]
    displacement=estimateMonomodalDeformationField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam,0,displacementList)
    warpPyramid=[rcommon.warpVolume(movingPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids3DCoronal(warpPyramid, fixedPyramid)
    sh=movingPyramid[0].shape
    rcommon.overlayImages(warpPyramid[0][:,sh[1]//4,:], fixedPyramid[0][:,sh[1]//4,:])
    rcommon.overlayImages(warpPyramid[0][:,sh[1]//2,:], fixedPyramid[0][:,sh[1]//2,:])
    rcommon.overlayImages(warpPyramid[0][:,3*sh[1]//4,:], fixedPyramid[0][:,3*sh[1]//4,:])
    rcommon.overlayImages(warpPyramid[0][sh[0]//4,:,:], fixedPyramid[0][sh[0]//4,:,:])
    rcommon.overlayImages(warpPyramid[0][sh[0]//2,:,:], fixedPyramid[0][sh[0]//2,:,:])
    rcommon.overlayImages(warpPyramid[0][3*sh[0]//4,:,:], fixedPyramid[0][3*sh[0]//4,:,:])
    rcommon.overlayImages(warpPyramid[0][:,:,sh[2]//4], fixedPyramid[0][:,:,sh[2]//4])
    rcommon.overlayImages(warpPyramid[0][:,:,sh[2]//2], fixedPyramid[0][:,:,sh[2]//2])
    rcommon.overlayImages(warpPyramid[0][:,:,3*sh[2]//4], fixedPyramid[0][:,:,3*sh[2]//4])
    #rcommon.plotDeformationField(displacement)
    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2 + displacement[...,2]**2)
    maxNorm=np.max(nrm)
    #displacement[...,0]*=(maskMoving + maskFixed)
    #displacement[...,1]*=(maskMoving + maskFixed)
    #rcommon.plotDeformationField(displacement)
    #figure()
    #imshow(nrm[:,sh[1]//2,:])
    print 'Max global displacement: ', maxNorm
    return displacementList, warpPyramid

###############################################################
####### Non-linear Multimodal registration - EM (2D)###########
###############################################################
def estimateNewMultimodalDeformationField2D(moving, fixed, lambdaDisplacement, quantizationLevels, maxOuterIter, previousDisplacement, useECQMMF):
    epsilon=1e-5
    sh=moving.shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]
    displacement     =np.empty(shape=(moving.shape)+(2,), dtype=np.float64)
    residuals=np.zeros(shape=(moving.shape), dtype=np.float64)
    gradientField    =np.empty(shape=(moving.shape)+(2,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
    fixedQ=None
    grayLevels=None
    if(useECQMMF):
        grayLevels, variances=ecqmmf.initialize_constant_models(fixed, quantizationLevels)
        fixedQ, grayLevels, variances, probs=ecqmmf.ecqmmf(fixed, quantizationLevels, 0.05, 0.01, 20, 50, 1e-5)
    else:
        fixedQ, grayLevels, hist=tf.quantizePositiveImageCYTHON(fixed, quantizationLevels)
    fixedQ=np.array(fixedQ, dtype=np.int32)
    finished=False
    outerIter=0
    maxDisplacement=None
    maxVariation=None
    maxResidual=0
    while((not finished) and (outerIter<maxOuterIter)):
        outerIter+=1
        #---E step---
        warped=ndimage.map_coordinates(moving, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1]], prefilter=True)
        movingMask=(moving>0)*1.0
        warpedMovingMask=ndimage.map_coordinates(movingMask, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1]], order=0, prefilter=False)
        warpedMovingMask=warpedMovingMask.astype(np.int32)
        means, variances=tf.computeMaskedImageClassStatsCYTHON(warpedMovingMask, warped, quantizationLevels, fixedQ)
        means[0]=0
        means=np.array(means)
        variances=np.array(variances)
        sigmaField=variances[fixedQ]
        deltaField=means[fixedQ]-warped
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
            maxVariation=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradientField,  lambdaDisplacement, totalDisplacement, displacement, residuals)
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
            plt.subplot(1,3,1)
            plt.imshow(means[fixedQ],cmap=plt.cm.gray)
            plt.title("Estimated warped modality")
            plt.subplot(1,3,2)
            plt.imshow(warpedMovingMask,cmap=plt.cm.gray)
            plt.title("Warped mask")
            plt.subplot(1,3,3)
            plt.plot(means)
            plt.title("Means")
    print "Iter: ",outerIter, "Max displacement:", maxDisplacement, "Max variation:",maxVariation, "Max residual:", maxResidual
    if(previousDisplacement!=None):
        return totalDisplacement-previousDisplacement
    return totalDisplacement

def estimateMultimodalDeformationField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level=0, displacementList=None, useECQMMF=True):
    n=len(movingPyramid)
    if(useECQMMF):
        quantizationLevels=64
    else:
        quantizationLevels=256//(2**level)
    #quantizationLevels=256
    if(level==(n-1)):
        displacement=estimateNewMultimodalDeformationField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, quantizationLevels, maxOuterIter[level], None, useECQMMF)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement
    subDisplacement=estimateMultimodalDeformationField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level+1, displacementList, useECQMMF)
    sh=movingPyramid[level].shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]*0.5
    upsampled=np.empty(shape=(movingPyramid[level].shape)+(2,), dtype=np.float64)
    upsampled[:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,0], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,1], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement=estimateNewMultimodalDeformationField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, quantizationLevels, maxOuterIter[level], upsampled, useECQMMF)
    newDisplacement+=upsampled
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return newDisplacement

def registerNonlinearMultimodal2D(moving, fixed, lambdaParam, levels):
    maskMoving=moving>0
    maskFixed=fixed>0
    movingPyramid=[img for img in rcommon.pyramid_gaussian_3D(moving, levels, maskMoving)]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_3D(fixed, levels, maskFixed)]
    maxOuterIter=[10,50,100,100,100,100]
    displacement=estimateMultimodalDeformationField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0, None, False)
    warped=rcommon.warpVolume(movingPyramid[0], displacement)
    return displacement, warped

def testEstimateMultimodalDeformationField2DMultiScale(lambdaParam, synthetic, useECQMMF):
    #fname0='IBSR_01_to_02.nii.gz'
    #fname1='data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz'
    displacementGTName='templateToIBSR01_GT.npy'
    fnameMoving='data/t2/IBSR_t2template_to_01.nii.gz'
    fnameFixed='data/t1/IBSR_template_to_01.nii.gz'
    nib_moving = nib.load(fnameMoving)
    nib_fixed = nib.load(fnameFixed)
    moving=nib_moving.get_data().squeeze().astype(np.float64)
    fixed=nib_fixed.get_data().squeeze().astype(np.float64)
    sl=moving.shape
    sr=fixed.shape    
    moving=moving[:,sl[1]//2,:].copy()
    fixed=fixed[:,sr[1]//2,:].copy()
    moving=(moving-moving.min())/(moving.max()-moving.min())
    fixed=(fixed-fixed.min())/(fixed.max()-fixed.min())
    maxOuterIter=[10,50,100,100,100,100]
    if(synthetic):
        print 'Generating synthetic field...'
        #----apply synthetic deformation field to fixed image
        GT=rcommon.createDeformationField_type2(fixed.shape[0], fixed.shape[1], 8)
        warpedFixed=rcommon.warpImage(fixed,GT)
    else:
        templateT1=nib.load('data/t1/IBSR_template_to_01.nii.gz')
        templateT1=templateT1.get_data().squeeze().astype(np.float64)
        sh=templateT1.shape
        templateT1=templateT1[:,sh[1]//2,:]
        templateT1=(templateT1-templateT1.min())/(templateT1.max()-templateT1.min())
        if(os.path.exists(displacementGTName)):
            print 'Loading precomputed realistic field...'
            GT=np.load(displacementGTName)
        else:
            print 'Generating realistic field...'
            #load two T1 images: the template and an IBSR sample
            ibsrT1=nib.load('data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz')
            ibsrT1=ibsrT1.get_data().squeeze().astype(np.float64)
            ibsrT1=ibsrT1[:,sh[1]//2,:]
            ibsrT1=(ibsrT1-ibsrT1.min())/(ibsrT1.max()-ibsrT1.min())
            #register the template(moving) to the ibsr sample(fixed)
            maskMoving=templateT1>0
            maskFixed=ibsrT1>0
            movingPyramid=[img for img in rcommon.pyramid_gaussian_2D(templateT1, 3, maskMoving)]
            fixedPyramid=[img for img in rcommon.pyramid_gaussian_2D(ibsrT1, 3, maskFixed)]
            #----apply 'realistic' deformation field to fixed image
            GT=estimateMultimodalDeformationField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0, None, useECQMMF)
            np.save(displacementGTName, GT)
        warpedFixed=rcommon.warpImage(templateT1, GT)
    print 'Registering T2 (template) to deformed T1 (template)...'
    level=3
    maskMoving=moving>0
    maskFixed=warpedFixed>0
    movingPyramid=[img for img in rcommon.pyramid_gaussian_2D(moving, level, maskMoving)]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_2D(warpedFixed, level, maskFixed)]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(moving, cmap=plt.cm.gray)
    plt.title('Moving')
    plt.subplot(1,2,2)
    plt.imshow(warpedFixed, cmap=plt.cm.gray)
    plt.title('Fixed')
    rcommon.plotOverlaidPyramids(movingPyramid, fixedPyramid)
    displacementList=[]
    displacement=estimateMultimodalDeformationField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0, displacementList, useECQMMF)
    warpPyramid=[rcommon.warpImage(movingPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids(warpPyramid, fixedPyramid)
    rcommon.overlayImages(warpPyramid[0], fixedPyramid[0])
    rcommon.plotDeformationField(displacement)
    displacement[...,0]*=(maskFixed)
    displacement[...,1]*=(maskFixed)
    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
    maxNorm=np.max(nrm)
    rcommon.plotDeformationField(displacement)
    residual=((displacement-GT))**2
    meanDisplacementError=np.sqrt(residual.sum(2)*(maskFixed)).mean()
    stdevDisplacementError=np.sqrt(residual.sum(2)*(maskFixed)).std()
    print 'Max global displacement: ', maxNorm
    print 'Mean displacement error: ', meanDisplacementError,'(',stdevDisplacementError,')'

###############################################################
####### Non-linear Multimodal registration - EM (3D)###########
###############################################################
def estimateNewMultimodalDeformationField3D(moving, fixed, lambdaParam, maxOuterIter, quantizationLevels, previousDisplacement=None):
    epsilon=1e-4
    sh=moving.shape
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
    residuals        =np.zeros(shape=(moving.shape),      dtype=np.float64)
    displacement     =np.empty(shape=(moving.shape)+(3,), dtype=np.float64)
    gradientField    =np.empty(shape=(moving.shape)+(3,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(moving.shape)+(3,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
    fixedQ, grayLevels, hist=tf.quantizePositiveVolumeCYTHON(fixed, quantizationLevels)
    fixedQ=np.array(fixedQ, dtype=np.int32)
    finished=False
    outerIter=0
    maxDisplacement=None
    maxVariation=None
    maxResidual=0
    while((not finished) and (outerIter<maxOuterIter)):
        outerIter+=1
        print "Outer:", outerIter
        #---E step---
        warped=ndimage.map_coordinates(moving, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1], X2+totalDisplacement[...,2]], prefilter=True)
        movingMask=(moving>0)*1.0
        warpedMovingMask=ndimage.map_coordinates(movingMask, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1], X2+totalDisplacement[...,2]], order=0, prefilter=False)
        warpedMovingMask=warpedMovingMask.astype(np.int32)
        means, variances=tf.computeMaskedVolumeClassStatsCYTHON(warpedMovingMask, warped, quantizationLevels, fixedQ)
        means[0]=0
        means=np.array(means)
        variances=np.array(variances)
        sigmaField=variances[fixedQ]
        deltaField=means[fixedQ]-warped
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
            plt.imshow(means[fixedQ[:,sh[1]//2,:]],cmap=plt.cm.gray)
            plt.title("Estimated warped modality")
            plt.subplot(1,2,2)
            plt.plot(means)
            plt.title("Means")
    print "Iter: ",outerIter, "Max displacement:", maxDisplacement, "Max variation:",maxVariation, "Max residual:", maxResidual
    if(previousDisplacement!=None):
        return totalDisplacement-previousDisplacement
    return totalDisplacement

def estimateMultimodalDeformationField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level=0, displacementList=None):
    n=len(movingPyramid)
    quantizationLevels=256//(2**level)
    if(level==(n-1)):
        displacement=estimateNewMultimodalDeformationField3D(movingPyramid[level], fixedPyramid[level], lambdaParam, maxOuterIter[level], quantizationLevels, None)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement
    subDisplacement=estimateMultimodalDeformationField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level+1, displacementList)
    sh=movingPyramid[level].shape
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]*0.5
    upsampled=np.empty(shape=(movingPyramid[level].shape)+(3,), dtype=np.float64)
    upsampled[:,:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,:,0], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,:,1], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,:,2]=ndimage.map_coordinates(subDisplacement[:,:,:,2], [X0, X1, X2], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement=estimateNewMultimodalDeformationField3D(movingPyramid[level], fixedPyramid[level], lambdaParam, maxOuterIter[level], quantizationLevels, upsampled)
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

def testEstimateMultimodalDeformationField3DMultiScale(lambdaParam=250, synthetic=False):
    displacementGTName='templateToIBSR01_GT3D.npy'
    fnameMoving='data/t2/IBSR_t2template_to_01.nii.gz'
    fnameFixed='data/t1/IBSR_template_to_01.nii.gz'
    pyramidMaxLevel=3
    maxDisplacement=2**pyramidMaxLevel
    nib_moving = nib.load(fnameMoving)
    nib_fixed = nib.load(fnameFixed)
    moving=nib_moving.get_data().squeeze().astype(np.float64)
    fixed=nib_fixed.get_data().squeeze().astype(np.float64)
    moving=(moving-moving.min())/(moving.max()-moving.min())
    fixed=(fixed-fixed.min())/(fixed.max()-fixed.min())
    maxOuterIter=[10,50,100,100,100,100]
    if(synthetic):
        print 'Generating synthetic field...'
        #----apply synthetic deformation field to fixed image
        GT=rcommon.createDeformationField3D_type2(fixed.shape, maxDisplacement)
        warpedFixed=rcommon.warpVolume(fixed,GT)
    else:
        templateT1=nib.load(fnameFixed)
        templateT1=templateT1.get_data().squeeze().astype(np.float64)
        templateT1=(templateT1-templateT1.min())/(templateT1.max()-templateT1.min())
        if(os.path.exists(displacementGTName)):
            print 'Loading precomputed realistic field...'
            GT=np.load(displacementGTName)
        else:
            print 'Generating realistic field...'
            #load two T1 images: the template and an IBSR sample
            ibsrT1=nib.load('data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz')
            ibsrT1=ibsrT1.get_data().squeeze().astype(np.float64)
            ibsrT1=(ibsrT1-ibsrT1.min())/(ibsrT1.max()-ibsrT1.min())
            #register the template(moving) to the ibsr sample(fixed)
            maskMoving=templateT1>0
            maskFixed=ibsrT1>0
            movingPyramid=[img for img in rcommon.pyramid_gaussian_3D(templateT1, pyramidMaxLevel, maskMoving)]
            fixedPyramid=[img for img in rcommon.pyramid_gaussian_3D(ibsrT1, pyramidMaxLevel, maskFixed)]
            #----apply 'realistic' deformation field to fixed image
            GT=estimateMultimodalDeformationField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0, None)
            np.save(displacementGTName, GT)
        warpedFixed=rcommon.warpVolume(templateT1, GT)
    print 'Registering T2 (template) to deformed T1 (template)...'
    maskMoving=moving>0
    maskFixed=warpedFixed>0
    movingPyramid=[img for img in rcommon.pyramid_gaussian_3D(moving, pyramidMaxLevel, maskMoving)]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_3D(warpedFixed, pyramidMaxLevel, maskFixed)]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(moving[:,moving.shape[1]//2,:], cmap=plt.cm.gray)
    plt.title('Moving')
    plt.subplot(1,2,2)
    plt.imshow(warpedFixed[:,warpedFixed.shape[1]//2,:], cmap=plt.cm.gray)
    plt.title('Fixed')
    rcommon.plotOverlaidPyramids3DCoronal(movingPyramid, fixedPyramid)
    displacementList=[]
    displacement=estimateMultimodalDeformationField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0, displacementList)
    warpPyramid=[rcommon.warpVolume(movingPyramid[i], displacementList[i]) for i in range(pyramidMaxLevel+1)]
    rcommon.plotOverlaidPyramids3DCoronal(warpPyramid, fixedPyramid)
    displacement[...,0]*=(maskFixed)
    displacement[...,1]*=(maskFixed)
    displacement[...,2]*=(maskFixed)
    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2 + displacement[...,2]**2)
    maxNorm=np.max(nrm)
    residual=((displacement-GT))**2
    meanDisplacementError=np.sqrt(residual.sum(3)*(maskFixed)).mean()
    stdevDisplacementError=np.sqrt(residual.sum(3)*(maskFixed)).std()
    print 'Max global displacement: ', maxNorm
    print 'Mean displacement error: ', meanDisplacementError,'(',stdevDisplacementError,')'

if __name__=="__main__":
    testEstimateMultimodalDeformationField2DMultiScale(250, False, True)
    #testEstimateMultimodalDeformationField3DMultiScale(250, False)    
