import numpy as np
import scipy as sp
import tensorFieldUtils as tf
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
import registrationCommon as rcommon
from registrationCommon import const_prefilter_map_coordinates
import os
###############################################################
####### Non-linear Monomodal registration - EM (2D)############
###############################################################
def estimateNewMonomodalDiffeomorphicField2D(moving, fixed, lambdaParam, maxOuterIter, previousDisplacement, previousDisplacementInverse):
    '''
    Warning: in the monomodal case, the parameter lambda must be significantly lower than in the multimodal case. Try lambdaParam=1,
    as opposed as lambdaParam=150 used in the multimodal case
    '''
    innerTolerance=1e-4
    outerTolerance=1e-3
    sh=moving.shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]
    displacement     =np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
    residuals=np.zeros(shape=(moving.shape), dtype=np.float64)
    gradientField    =np.empty(shape=(moving.shape)+(2,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
    totalDisplacementInverse=np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
        totalDisplacementInverse[...]=previousDisplacementInverse
    outerIter=0
    framesToCapture=5
    maxOuterIter=framesToCapture*((maxOuterIter+framesToCapture-1)/framesToCapture)
    itersPerCapture=maxOuterIter/framesToCapture
    plt.figure()
    while(outerIter<maxOuterIter):
        outerIter+=1
        print 'Outer iter:', outerIter
        warped=ndimage.map_coordinates(moving, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1]], prefilter=const_prefilter_map_coordinates)
        if((outerIter==1) or (outerIter%itersPerCapture==0)):
            plt.subplot(1,framesToCapture+1, 1+outerIter/itersPerCapture)
            rcommon.overlayImages(warped, fixed, False)
            plt.title('Iter:'+str(outerIter-1))
        sigmaField=np.ones_like(warped, dtype=np.float64)
        deltaField=fixed-warped
        g0, g1=sp.gradient(warped)
        gradientField[:,:,0]=g0
        gradientField[:,:,1]=g1
        maxVariation=1+innerTolerance
        innerIter=0
        maxResidual=0
        displacement[...]=0
        maxInnerIter=1000
        while((maxVariation>innerTolerance)and(innerIter<maxInnerIter)):
            innerIter+=1
            maxVariation=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, totalDisplacement, displacement, residuals)
            opt=np.max(residuals)
            if(maxResidual<opt):
                maxResidual=opt
        maxDisplacement=np.max(np.abs(displacement))
        expd, invexpd=tf.vector_field_exponential(displacement)
        totalDisplacement=tf.compose_vector_fields(expd, totalDisplacement)
        totalDisplacementInverse=tf.compose_vector_fields(totalDisplacementInverse, invexpd)
        if(maxDisplacement<outerTolerance):
            break
    print "Iter: ",innerIter, "Max lateral displacement:", maxDisplacement, "Max variation:",maxVariation, "Max residual:", maxResidual
    if(previousDisplacement!=None):
        return totalDisplacement-previousDisplacement, totalDisplacementInverse
    return totalDisplacement, totalDisplacementInverse


def estimateMonomodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level, displacementList):
    n=len(movingPyramid)
    if(level==(n-1)):
        #displacement=estimateNewMonomodalDeformationField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, maxOuterIter[level], None)
        displacement, inverse=estimateNewMonomodalDiffeomorphicField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, maxOuterIter[level], None, None)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement, inverse
    subDisplacement, subDisplacementInverse=estimateMonomodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level+1, displacementList)
    sh=movingPyramid[level].shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]*0.5
    upsampled=np.empty(shape=(movingPyramid[level].shape)+(2,), dtype=np.float64)
    upsampled[:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,0], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,1], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampledInverse=np.empty(shape=(movingPyramid[level].shape)+(2,), dtype=np.float64)
    upsampledInverse[:,:,0]=ndimage.map_coordinates(subDisplacementInverse[:,:,0], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampledInverse[:,:,1]=ndimage.map_coordinates(subDisplacementInverse[:,:,1], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement, newDisplacementInverse=estimateNewMonomodalDiffeomorphicField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, maxOuterIter[level], upsampled, upsampledInverse)
    newDisplacement+=upsampled
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return np.array(newDisplacement), np.array(newDisplacementInverse)

def testEstimateMonomodalDiffeomorphicField2DMultiScale(lambdaParam):
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
    maxIter=200
    displacement, inverse=estimateMonomodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxIter, 0,displacementList)
    residual=tf.compose_vector_fields(displacement, inverse)
    warpPyramid=[rcommon.warpImage(movingPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids(warpPyramid, fixedPyramid)
    rcommon.overlayImages(warpPyramid[0], fixedPyramid[0])
    rcommon.plotDiffeomorphism(displacement, inverse, residual)

def testCircleToCMonomodalDiffeomorphic(lambdaParam):
    fname0='data/circle.png'
    #fname0='/home/omar/Desktop/C_trans.png'
    fname1='data/C.png'
    nib_moving=plt.imread(fname0)
    nib_fixed=plt.imread(fname1)
    moving=nib_moving[:,:,0]
    fixed=nib_fixed[:,:,1]
    moving=(moving-moving.min())/(moving.max() - moving.min())
    fixed=(fixed-fixed.min())/(fixed.max() - fixed.min())
    level=3
    maskMoving=moving>0
    maskFixed=fixed>0
#    movingPyramid=[img for img in rcommon.pyramid_gaussian_2D(moving, level, maskMoving)]
#    fixedPyramid=[img for img in rcommon.pyramid_gaussian_2D(fixed, level, maskFixed)]
    movingPyramid=[img for img in rcommon.pyramid_gaussian_2D(moving, level, np.ones_like(maskMoving))]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_2D(fixed, level, np.ones_like(maskFixed))]
    rcommon.plotOverlaidPyramids(movingPyramid, fixedPyramid)
    displacementList=[]
    maxOuterIter=[10,50,100,100,100,100,100,100,100]
    displacement, inverse=estimateMonomodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0,displacementList)
    directInverse=tf.invert_vector_field(displacement, 1, 1000, 1e-7)
    residual=np.array(tf.compose_vector_fields(displacement, inverse))
    directResidual=np.array(tf.compose_vector_fields(displacement, directInverse))
    warpPyramid=[rcommon.warpImage(movingPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids(warpPyramid, fixedPyramid)
    rcommon.overlayImages(warpPyramid[0], fixedPyramid[0])
#    displacement[...,0]*=(maskMoving + maskFixed)
#    displacement[...,1]*=(maskMoving + maskFixed)
    rcommon.plotDiffeomorphism(displacement, inverse, residual, 'inv-joint', 7)
    rcommon.plotDiffeomorphism(displacement, directInverse, directResidual, 'inv-direct', 7)

###############################################################
####### Diffeomorphic Multimodal registration - EM (2D)########
###############################################################
def estimateNewMultimodalDiffeomorphicField2D(moving, fixed, lambdaDisplacement, quantizationLevels, maxOuterIter, previousDisplacement, previousDisplacementInverse):
    innerTolerance=1e-4
    outerTolerance=1e-3
    sh=moving.shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]
    displacement     =np.empty(shape=(moving.shape)+(2,), dtype=np.float64)
    residuals=np.zeros(shape=(moving.shape), dtype=np.float64)
    gradientField    =np.empty(shape=(moving.shape)+(2,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
    totalDisplacementInverse=np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
        totalDisplacementInverse[...]=previousDisplacementInverse
    fixedQ=None
    grayLevels=None
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
        movingMask=((moving>0)*1.0)*((fixed>0)*1.0)
        warpedMovingMask=ndimage.map_coordinates(movingMask, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1]], order=0, prefilter=False)
        warpedMovingMask=warpedMovingMask.astype(np.int32)
        means, variances=tf.computeMaskedImageClassStatsCYTHON(warpedMovingMask, warped, quantizationLevels, fixedQ)            
        means[0]=0
        means=np.array(means)
        variances=np.array(variances)
        sigmaField=variances[fixedQ]
        deltaField=means[fixedQ]-warped#########Delta-field using Arce's rule
        #--M step--
        g0, g1=sp.gradient(warped)
        gradientField[:,:,0]=g0
        gradientField[:,:,1]=g1
        maxVariation=1+innerTolerance
        innerIter=0
        maxInnerIter=1000
        displacement[...]=0
        while((maxVariation>innerTolerance)and(innerIter<maxInnerIter)):
            innerIter+=1
            maxVariation=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradientField,  lambdaDisplacement, totalDisplacement, displacement, residuals)
            opt=np.max(residuals)
            if(maxResidual<opt):
                maxResidual=opt
        #--accumulate displacement--
        expd, invexpd=tf.vector_field_exponential(displacement)
        totalDisplacement=tf.compose_vector_fields(expd, totalDisplacement)
        totalDisplacementInverse=tf.compose_vector_fields(totalDisplacementInverse, invexpd)
        #--check stop condition--
        nrm=np.sqrt(displacement[...,0]**2+displacement[...,1]**2)
        #maxDisplacement=np.max(nrm)
        maxDisplacement=np.mean(nrm)
        if((maxDisplacement<outerTolerance)or(outerIter>=maxOuterIter)):
            finished=True
#            plt.figure()
#            plt.subplot(1,3,1)
#            plt.imshow(means[fixedQ],cmap=plt.cm.gray)    
#            plt.title("Estimated warped modality")
#            plt.subplot(1,3,2)
#            plt.imshow(fixedQ,cmap=plt.cm.gray)
#            plt.title("Quantized")
#            plt.subplot(1,3,3)
#            plt.plot(means)
#            plt.title("Means")
    print "Iter: ",outerIter, "Mean displacement:", maxDisplacement, "Max variation:",maxVariation, "Max residual:", maxResidual
    if(previousDisplacement!=None):
        return totalDisplacement-previousDisplacement, totalDisplacementInverse
    return totalDisplacement, totalDisplacementInverse

def estimateMultimodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level=0, displacementList=None):
    n=len(movingPyramid)
    quantizationLevels=256
    if(level==(n-1)):
        displacement, inverse=estimateNewMultimodalDiffeomorphicField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, quantizationLevels, maxOuterIter[level], None, None)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement, inverse
    subDisplacement, subDisplacementInverse=estimateMultimodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level+1, displacementList)
    sh=movingPyramid[level].shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]*0.5
    upsampled=np.empty(shape=(movingPyramid[level].shape)+(2,), dtype=np.float64)
    upsampled[:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,0], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,1], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampledInverse=np.empty(shape=(movingPyramid[level].shape)+(2,), dtype=np.float64)
    upsampledInverse[:,:,0]=ndimage.map_coordinates(subDisplacementInverse[:,:,0], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampledInverse[:,:,1]=ndimage.map_coordinates(subDisplacementInverse[:,:,1], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement, inverse=estimateNewMultimodalDiffeomorphicField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, quantizationLevels, maxOuterIter[level], upsampled, upsampledInverse)
    newDisplacement+=upsampled
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return newDisplacement, inverse

def runArcesExperiment(rootDir, lambdaParam, maxOuterIter):
    #---Load displacement field---
    dxName=rootDir+'Vx.dat'
    dyName=rootDir+'Vy.dat'
    dx=np.loadtxt(dxName)
    dy=np.loadtxt(dyName)
    GT_in=np.ndarray(shape=dx.shape+(2,), dtype=np.float64)
    GT_in[...,0]=dy
    GT_in[...,1]=dx
    GT, GTinv=tf.vector_field_exponential(GT_in)
    GTres=tf.compose_vector_fields(GT, GTinv)
    #---Load input images---
    fnameT1=rootDir+'t1.jpg'
    fnameT2=rootDir+'t2.jpg'
    fnamePD=rootDir+'pd.jpg'
    fnameMask=rootDir+'Mascara.bmp'
    t1=plt.imread(fnameT1)[...,0].astype(np.float64)
    t2=plt.imread(fnameT2)[...,0].astype(np.float64)
    pd=plt.imread(fnamePD)[...,0].astype(np.float64)
    t1=(t1-t1.min())/(t1.max()-t1.min())
    t2=(t2-t2.min())/(t2.max()-t2.min())
    pd=(pd-pd.min())/(pd.max()-pd.min())
    mask=plt.imread(fnameMask).astype(np.float64)
    fixed=t1
    moving=t2
    maskMoving=mask>0
    maskFixed=mask>0
    fixed*=mask
    moving*=mask
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(t1, cmap=plt.cm.gray)
    plt.title('Input T1')
    plt.subplot(1,4,2)
    plt.imshow(t2, cmap=plt.cm.gray)
    plt.title('Input T2')
    plt.subplot(1,4,3)
    plt.imshow(pd, cmap=plt.cm.gray)
    plt.title('Input PD')
    plt.subplot(1,4,4)
    plt.imshow(mask, cmap=plt.cm.gray)
    plt.title('Input Mask')
    #-------------------------
    warpedFixed=rcommon.warpImage(fixed,GT)
    print 'Registering T2 (template) to deformed T1 (template)...'
    level=3
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
    displacement, inverse=estimateMultimodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0, displacementList)
    residual=tf.compose_vector_fields(displacement, inverse)
    warpPyramid=[rcommon.warpImage(movingPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids(warpPyramid, fixedPyramid)
    rcommon.overlayImages(warpPyramid[0], fixedPyramid[0])
    displacement[...,0]*=(maskFixed)
    displacement[...,1]*=(maskFixed)
    #----plot deformations---
    rcommon.plotDiffeomorphism(GT, GTinv, GTres, 7)
    rcommon.plotDiffeomorphism(displacement, inverse, residual, 7)
    #----statistics---
    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
    nrm*=maskFixed
    maxNorm=np.max(nrm)
    residual=((displacement-GT))**2
    meanDisplacementError=np.sqrt(residual.sum(2)*(maskFixed)).mean()
    stdevDisplacementError=np.sqrt(residual.sum(2)*(maskFixed)).std()
    print 'Max global displacement: ', maxNorm
    print 'Mean displacement error: ', meanDisplacementError,'(',stdevDisplacementError,')'

def runAllArcesExperiments(lambdaParam, maxOuterIter):
    rootDirs=['/opt/registration/data/arce/GT01/',
              '/opt/registration/data/arce/GT02/',
              '/opt/registration/data/arce/GT03/',
              '/opt/registration/data/arce/GT04/']
#    rootDirs=['/opt/registration/data/arce/GT02/']
    for rootDir in rootDirs:
        runArcesExperiment(rootDir, lambdaParam, maxOuterIter)
    print 'done.'

def testInversion(lambdaParam):
    fname0='data/circle.png'
    fname1='data/C.png'
    circleToCDisplacementName='circleToCDisplacement.npy'
    circleToCDisplacementInverseName='circleToCDisplacementInverse.npy'
    nib_moving=plt.imread(fname0)
    nib_fixed=plt.imread(fname1)
    moving=nib_moving[:,:,0]
    fixed=nib_fixed[:,:,1]
    moving=(moving-moving.min())/(moving.max() - moving.min())
    fixed=(fixed-fixed.min())/(fixed.max() - fixed.min())
    level=3
    maskMoving=moving>0
    maskFixed=fixed>0
    movingPyramid=[img for img in rcommon.pyramid_gaussian_2D(moving, level, np.ones_like(maskMoving))]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_2D(fixed, level, np.ones_like(maskFixed))]
    rcommon.plotOverlaidPyramids(movingPyramid, fixedPyramid)
    displacementList=[]
    maxOuterIter=[10,50,100,100,100,100,100,100,100]
    if(os.path.exists(circleToCDisplacementName)):
        displacement=np.load(circleToCDisplacementName)
        inverse=np.load(circleToCDisplacementInverseName)
    else:
        displacement, inverse=estimateMonomodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0,displacementList)
        np.save(circleToCDisplacementName, displacement)
        np.save(circleToCDisplacementInverseName, inverse)
    print 'vector field exponential'
    expd, invexpd=tf.vector_field_exponential(displacement)
    print 'vector field inversion'
    directInverse=tf.invert_vector_field(displacement, 1.0, 10000, 1e-7)
    print 'vector field inversion'
    directExpInverse=tf.invert_vector_field(expd, 1.0, 10000, 1e-7)
    ###Now compare inversions###
    residualJoint=np.array(tf.compose_vector_fields(displacement, inverse))
    residualDirect=np.array(tf.compose_vector_fields(displacement, directInverse))
    residualExpJoint=np.array(tf.compose_vector_fields(expd, invexpd))
    residualExpDirect=np.array(tf.compose_vector_fields(expd, directExpInverse))
    rcommon.plotDiffeomorphism(displacement, inverse, residualJoint, 'D-joint', 7)
    rcommon.plotDiffeomorphism(expd, invexpd, residualExpJoint, 'expD-joint', 7)
    rcommon.plotDiffeomorphism(displacement, directInverse, residualDirect, 'D-direct', 7)
    rcommon.plotDiffeomorphism(expd, directExpInverse, residualExpDirect, 'expD-direct', 7)
    
if __name__=='__main__':
#    testInversion(5)
    testCircleToCMonomodalDiffeomorphic(5)
    #######################################
#    maxOuterIter=[500,500,500,500,500,500]
#    runAllArcesExperiments(2000, maxOuterIter)
