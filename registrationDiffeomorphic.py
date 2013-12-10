import numpy as np
import scipy as sp
import tensorFieldUtils as tf
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
import registrationCommon as rcommon
from registrationCommon import const_prefilter_map_coordinates
import os
import sys
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
    moving=nib_moving.get_data().squeeze().astype(np.float64)
    fixed=nib_fixed.get_data().squeeze().astype(np.float64)
    moving=np.copy(moving, order='C')
    fixed=np.copy(fixed, order='C')
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
    import numpy as np
    import tensorFieldUtils as tf
    import matplotlib.pyplot as plt
    import registrationCommon as rcommon
    fname0='data/circle.png'
    #fname0='data/C_trans.png'
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
    X1,X0=np.mgrid[0:displacement.shape[0], 0:displacement.shape[1]]
    detJacobian=rcommon.computeJacobianField(displacement)
    plt.figure()
    plt.imshow(detJacobian)
    CS=plt.contour(X0,X1,detJacobian,levels=[0.0], colors='b')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('det(J(displacement))')
    print 'J range:', '[', detJacobian.min(), detJacobian.max(),']'
    directInverse=tf.invert_vector_field(displacement, 0.5, 1000, 1e-7)
    detJacobianInverse=rcommon.computeJacobianField(directInverse)
    plt.figure()
    plt.imshow(detJacobianInverse)
    CS=plt.contour(X0,X1,detJacobianInverse, levels=[0.0],colors='w')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('det(J(displacement^-1))')
    print 'J^-1 range:', '[', detJacobianInverse.min(), detJacobianInverse.max(),']'
    #directInverse=rcommon.invert_vector_field_fixed_point(displacement, 1000, 1e-7)
    residual=np.array(tf.compose_vector_fields(displacement, inverse))
    directResidual=np.array(tf.compose_vector_fields(displacement, directInverse))
#    warpPyramid=[rcommon.warpImage(movingPyramid[i], displacementList[i]) for i in range(level+1)]
#    rcommon.plotOverlaidPyramids(warpPyramid, fixedPyramid)
#    rcommon.overlayImages(warpPyramid[0], fixedPyramid[0])
    rcommon.plotDiffeomorphism(displacement, inverse, residual, 'inv-joint', 7)
    rcommon.plotDiffeomorphism(displacement, directInverse, directResidual, 'inv-direct', 7)
    tf.write_double_buffer(displacement.reshape(-1), 'displacement.bin')
















def displayRegistrationResultDiff():
    fnameMoving='data/affineRegistered/templateT1ToIBSR01T1.nii.gz'
    fnameFixed='data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz'
    nib_fixed = nib.load(fnameFixed)
    fixed=nib_fixed.get_data().squeeze()
    fixed=np.copy(fixed,order='C')
    nib_moving = nib.load(fnameMoving)
    moving=nib_moving.get_data().squeeze()
    moving=np.copy(moving, order='C')
    fnameDisplacement='displacement_templateT1ToIBSR01T1_diffMulti.npy'
    fnameWarped='warped_templateT1ToIBSR01T1_diffMulti.npy'
    displacement=np.load(fnameDisplacement)
    warped=np.load(fnameWarped)
    sh=moving.shape
    shown=warped
    f=rcommon.overlayImages(shown[:,sh[1]//4,:], fixed[:,sh[1]//4,:])
    f=rcommon.overlayImages(shown[:,sh[1]//2,:], fixed[:,sh[1]//2,:])
    f=rcommon.overlayImages(shown[:,3*sh[1]//4,:], fixed[:,3*sh[1]//4,:])
    f=rcommon.overlayImages(shown[sh[0]//4,:,:], fixed[sh[0]//4,:,:])
    f=rcommon.overlayImages(shown[sh[0]//2,:,:], fixed[sh[0]//2,:,:])
    f=rcommon.overlayImages(shown[3*sh[0]//4,:,:], fixed[3*sh[0]//4,:,:])
    f=rcommon.overlayImages(shown[:,:,sh[2]//4], fixed[:,:,sh[2]//4])
    f=rcommon.overlayImages(shown[:,:,sh[2]//2], fixed[:,:,sh[2]//2])
    f=rcommon.overlayImages(shown[:,:,3*sh[2]//4], fixed[:,:,3*sh[2]//4])
    del f
    del displacement

###############################################################
####### Diffeomorphic Monomodal registration - EM (3D)#########
###############################################################
def estimateNewMonomodalDiffeomorphicField3D(moving, fixed, lambdaParam, maxOuterIter, previousDisplacement, reportProgress=False):
    '''
    Warning: in the monomodal case, the parameter lambda must be significantly lower than in the multimodal case. Try lambdaParam=1,
    as opposed as lambdaParam=150 used in the multimodal case
    '''
    innerTolerance=1e-3
    outerTolerance=1e-3
    displacement     =np.zeros(shape=(moving.shape)+(3,), dtype=np.float64)
    residuals=np.zeros(shape=(moving.shape), dtype=np.float64)
    gradientField    =np.empty(shape=(moving.shape)+(3,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(moving.shape)+(3,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
    outerIter=0
    while(outerIter<maxOuterIter):
        outerIter+=1
        if(reportProgress):
            print 'Iter:',outerIter,'/',maxOuterIter
        warped=np.array(tf.warp_volume(moving, totalDisplacement))
        sigmaField=np.ones_like(warped, dtype=np.float64)
        deltaField=fixed-warped
        g0, g1, g2=sp.gradient(warped)
        gradientField[:,:,:,0]=g0
        gradientField[:,:,:,1]=g1
        gradientField[:,:,:,2]=g2
        maxVariation=1+innerTolerance
        innerIter=0
        maxResidual=0
        displacement[...]=0
        maxInnerIter=50
        while((maxVariation>innerTolerance)and(innerIter<maxInnerIter)):
            innerIter+=1
            maxVariation=tf.iterateDisplacementField3DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, totalDisplacement, displacement, residuals)
            opt=np.max(residuals)
            if(maxResidual<opt):
                maxResidual=opt
        maxDisplacement=np.max(np.abs(displacement))
        totalDisplacement=tf.compose_vector_fields3D(displacement, totalDisplacement)
        if(maxDisplacement<outerTolerance):
            break
    print "Iter: ",outerIter, "Max lateral displacement:", maxDisplacement, "Max variation:",maxVariation, "Max residual:", maxResidual
    if(previousDisplacement!=None):
        return totalDisplacement-previousDisplacement
    return totalDisplacement

def estimateMonomodalDiffeomorphicField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level, displacementList):
    n=len(movingPyramid)
    if(level==(n-1)):
        displacement=estimateNewMonomodalDiffeomorphicField3D(movingPyramid[level], fixedPyramid[level], lambdaParam, maxOuterIter[level], None, level==0)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement
    subDisplacement=estimateMonomodalDiffeomorphicField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level+1, displacementList)
    sh=np.array(movingPyramid[level].shape)
    upsampled=np.array(tf.upsample_displacement_field3D(subDisplacement, sh))*2
    newDisplacement=estimateNewMonomodalDiffeomorphicField3D(movingPyramid[level], fixedPyramid[level], lambdaParam, maxOuterIter[level], upsampled, level==0)
    newDisplacement+=upsampled
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return np.array(newDisplacement)

def testEstimateMonomodalDiffeomorphicField3DMultiScale(lambdaParam):
    fnameMoving='data/affineRegistered/templateT1ToIBSR01T1.nii.gz'
    fnameFixed='data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz'
    moving = nib.load(fnameMoving)
    fixed= nib.load(fnameFixed)
    moving=moving.get_data().squeeze().astype(np.float64)
    fixed=fixed.get_data().squeeze().astype(np.float64)
    moving=np.copy(moving, order='C')
    fixed=np.copy(fixed, order='C')
    moving=(moving-moving.min())/(moving.max()-moving.min())
    fixed=(fixed-fixed.min())/(fixed.max()-fixed.min())
    level=3
    #maskMoving=np.ones_like(moving)
    #maskFixed=np.ones_like(fixed)
    movingPyramid=[img for img in rcommon.pyramid_gaussian_3D(moving, level, np.ones_like(moving))]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_3D(fixed, level, np.ones_like(fixed))]
    rcommon.plotOverlaidPyramids3DCoronal(movingPyramid, fixedPyramid)
    #maxOuterIter=[100,100,100,100,100,100,100,100,100]
    maxOuterIter=[3,3,3,3,3,3,3,3,3]
    #maxOuterIter=[10,20,50,100, 100, 100]
    displacement=estimateMonomodalDiffeomorphicField3DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0,None)
    warped=tf.warp_volume(movingPyramid[0], displacement)
    np.save('displacement_templateT1ToIBSR01T1_diff.npy', displacement)
    np.save('warped_templateT1ToIBSR01T1_diff.npy', warped)











###############################################################
####### Diffeomorphic Multimodal registration - EM (3D)########
###############################################################
def estimateNewMultimodalDiffeomorphicField3D(moving, fixed, initAffine, lambdaDisplacement, quantizationLevels, maxOuterIter, previousDisplacement, reportProgress=False):
    innerTolerance=1e-3
    outerTolerance=1e-3
    displacement     =np.empty(shape=(fixed.shape)+(3,), dtype=np.float64)
    residuals=np.zeros(shape=(fixed.shape), dtype=np.float64)
    gradientField    =np.empty(shape=(fixed.shape)+(3,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(fixed.shape)+(3,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
    fixedQ=None
    grayLevels=None
    fixedQ, grayLevels, hist=tf.quantizePositiveVolumeCYTHON(fixed, quantizationLevels)
    fixedQ=np.array(fixedQ, dtype=np.int32)
    finished=False
    outerIter=0
    maxDisplacement=None
    maxVariation=None
    maxResidual=0
    fixedMask=(fixed>0).astype(np.int32)
    movingMask=(moving>0).astype(np.int32)
    trustRegion=fixedMask*np.array(tf.warp_discrete_volumeNNAffine(movingMask, np.array(fixedMask.shape), initAffine))#consider only the overlap after affine registration
    while((not finished) and (outerIter<maxOuterIter)):
        outerIter+=1
        if(reportProgress):
            print 'Iter:',outerIter,'/',maxOuterIter
            #sys.stdout.flush()
        #---E step---
        #print "Warping..."
        #sys.stdout.flush()
        warped=np.array(tf.warp_volume(moving, totalDisplacement, initAffine))
        warpedMask=np.array(tf.warp_discrete_volumeNN(trustRegion, totalDisplacement, np.eye(4))).astype(np.int32)#the affine mapping was already applied
        #print "Warping NN..."
        #sys.stdout.flush()
        #warpedMovingMask=np.array(tf.warp_volumeNN(movingMask, totalDisplacement)).astype(np.int32)
        #print "Class stats..."
        #sys.stdout.flush()
        means, variances=tf.computeMaskedVolumeClassStatsCYTHON(warpedMask, warped, quantizationLevels, fixedQ)        
        means[0]=0
        means=np.array(means)
        variances=np.array(variances)
        sigmaField=variances[fixedQ]
        deltaField=means[fixedQ]-warped#########Delta-field using Arce's rule
        #--M step--
        g0, g1, g2=sp.gradient(warped)
        gradientField[:,:,:,0]=g0
        gradientField[:,:,:,1]=g1
        gradientField[:,:,:,2]=g2
        maxVariation=1+innerTolerance
        innerIter=0
        maxInnerIter=100
        displacement[...]=0
        #print "Iterating..."
        #sys.stdout.flush()
        while((maxVariation>innerTolerance)and(innerIter<maxInnerIter)):
            innerIter+=1
            maxVariation=tf.iterateDisplacementField3DCYTHON(deltaField, sigmaField, gradientField,  lambdaDisplacement, totalDisplacement, displacement, residuals)
            opt=np.max(residuals)
            if(maxResidual<opt):
                maxResidual=opt
        #--accumulate displacement--
        #print "Exponential3D. Range D:", displacement.min(), displacement.max()
        #sys.stdout.flush()
        expd, inverseNone=tf.vector_field_exponential3D(displacement, False)
        expd=np.array(expd)
        #print "Range expd:", expd.min(), expd.max(), "Range TD:", totalDisplacement.min(), totalDisplacement.max()
        #print "Compose vector fields..."
        #sys.stdout.flush()
        totalDisplacement=np.array(tf.compose_vector_fields3D(expd, totalDisplacement))
        #print "Composed rage:", totalDisplacement.min(), totalDisplacement.max()
        #sys.stdout.flush()
        #--check stop condition--
        nrm=np.sqrt(displacement[...,0]**2+displacement[...,1]**2+displacement[...,2]**2)
        #maxDisplacement=np.max(nrm)
        maxDisplacement=np.mean(nrm)
        if((maxDisplacement<outerTolerance)or(outerIter>=maxOuterIter)):
            finished=True
    print "Iter: ",outerIter, "Mean displacement:", maxDisplacement, "Max variation:",maxVariation, "Max residual:", maxResidual
    #sh=fixed.shape
    #rcommon.overlayImages(warped[:,sh[1]//2,:], fixed[:,sh[1]//2,:])
    #rcommon.overlayImages(warped[:,sh[1]//2,:]*warpedMask[:,sh[1]//2,:], fixed[:,sh[1]//2,:])
    #sys.stdout.flush()
    if(previousDisplacement!=None):
        #print 'Range TD:', totalDisplacement.min(), totalDisplacement.max(),'. Range PD:', previousDisplacement.min(), previousDisplacement.max()
        #sys.stdout.flush()
        return totalDisplacement-previousDisplacement
    return totalDisplacement

def estimateMultimodalDiffeomorphicField3DMultiScale(movingPyramid, fixedPyramid, initAffine, lambdaParam, maxOuterIter, level=0, displacementList=None):
    n=len(movingPyramid)
    quantizationLevels=256
    if(level==(n-1)):
        displacement=estimateNewMultimodalDiffeomorphicField3D(movingPyramid[level], fixedPyramid[level], initAffine, lambdaParam, quantizationLevels, maxOuterIter[level], None, level==0)
        if(displacementList!=None):
            displacementList.insert(0, displacement)
        return displacement
    subAffine=initAffine.copy()
    #subAffine=initAffine.copy()*0.5
    subAffine[:3,3]*=0.5
    subDisplacement=estimateMultimodalDiffeomorphicField3DMultiScale(movingPyramid, fixedPyramid, subAffine, lambdaParam, maxOuterIter, level+1, displacementList)
    sh=np.array(fixedPyramid[level].shape).astype(np.int32)
    upsampled=np.array(tf.upsample_displacement_field3D(subDisplacement, sh))*2
    newDisplacement=estimateNewMultimodalDiffeomorphicField3D(movingPyramid[level], fixedPyramid[level], initAffine, lambdaParam, quantizationLevels, maxOuterIter[level], upsampled, level==0)
    newDisplacement+=upsampled
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return newDisplacement

def testEstimateMultimodalDiffeomorphicField3DMultiScale(fnameMoving, fnameFixed, fnameAffine, lambdaParam):
    '''
        testEstimateMultimodalDiffeomorphicField3DMultiScale('IBSR_01_ana_strip.nii.gz', 't1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 'IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt', 100)
    '''
    print 'Registering', fnameMoving, 'to', fnameFixed,'with lambda=',lambdaParam  
    sys.stdout.flush()
    moving = nib.load(fnameMoving)
    fixed= nib.load(fnameFixed)
    M=moving.get_affine()
    F=fixed.get_affine()
    if not fnameAffine:
        T=np.eye(4)
    else:
        T=rcommon.readAntsAffine(fnameAffine)
    initAffine=np.linalg.inv(M).dot(T.dot(F))
    print initAffine
    moving=moving.get_data().squeeze().astype(np.float64)
    fixed=fixed.get_data().squeeze().astype(np.float64)
    moving=np.copy(moving, order='C')
    fixed=np.copy(fixed, order='C')
    moving=(moving-moving.min())/(moving.max()-moving.min())
    fixed=(fixed-fixed.min())/(fixed.max()-fixed.min())
    level=3
    maskMoving=moving>0
    maskFixed=fixed>0
    movingPyramid=[img for img in rcommon.pyramid_gaussian_3D(moving, level, maskMoving)]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_3D(fixed, level, maskFixed)]
    #maxOuterIter=[100,100,100,100,100,100,100,100,100]
    #maxOuterIter=[3,3,3,3,3,3,3,3,3]
    maxOuterIter=[10,20,50,100, 100, 100]
    displacement=estimateMultimodalDiffeomorphicField3DMultiScale(movingPyramid, fixedPyramid, initAffine, lambdaParam, maxOuterIter, 0,None)
    tf.prepend_affine_to_displacement_field(displacement, initAffine)
    warped=np.array(tf.warp_volume(movingPyramid[0], displacement, np.eye(4)))
    baseMoving=rcommon.getBaseFileName(fnameMoving)
    baseFixed=rcommon.getBaseFileName(fnameFixed)
    np.save('dispDiff_'+baseMoving+'_'+baseFixed+'.npy', displacement)
    imgWarped=nib.Nifti1Image(warped, np.eye(4))
    imgWarped.to_filename('warpedDiff_'+baseMoving+'_'+baseFixed+'.nii.gz')
    print 'Computing inverse...'
    lambdaParam=0.9
    maxIter=100
    tolerance=1e-4
    inverse=np.array(tf.invert_vector_field3D(displacement, lambdaParam, maxIter, tolerance))
    np.save('invdispDiff_'+baseMoving+'_'+baseFixed+'.npy', inverse)
    print 'Computing inversion error...'
    residual=np.array(tf.compose_vector_fields3D(displacement, inverse))
    np.save('resdispDiff_'+baseMoving+'_'+baseFixed+'.npy', residual)
    residual=np.sqrt(np.sum(residual**2,3))
    print "Mean residual norm:", residual.mean()," (",residual.std(), "). Max residual norm:", residual.max()
















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
    residualJoint=np.array(tf.compose_vector_fields(displacement, inverse)[0])
    residualDirect=np.array(tf.compose_vector_fields(displacement, directInverse)[0])
    residualExpJoint=np.array(tf.compose_vector_fields(expd, invexpd)[0])
    residualExpDirect=np.array(tf.compose_vector_fields(expd, directExpInverse)[0])
    rcommon.plotDiffeomorphism(displacement, inverse, residualJoint, 'D-joint')
    rcommon.plotDiffeomorphism(expd, invexpd, residualExpJoint, 'expD-joint')
    d,invd,res,jacobian=rcommon.plotDiffeomorphism(displacement, directInverse, residualDirect, 'D-direct')
    rcommon.plotDiffeomorphism(expd, directExpInverse, residualExpDirect, 'expD-direct')
    sp.misc.imsave('circleToC_deformation.png', d)
    sp.misc.imsave('circleToC_inverse_deformation.png', invd)
    sp.misc.imsave('circleToC_residual_deformation.png', res)
    tf.write_double_buffer(np.array(displacement).reshape(-1), '../inverse/experiments/displacement.bin')
    tf.write_double_buffer(np.array(displacement).reshape(-1), '../inverse/experiments/displacement_clean.bin')

def testInversion_invertible():
    displacement_clean=tf.create_invertible_displacement_field(256, 256, 0.5, 8)
    detJacobian=rcommon.computeJacobianField(displacement_clean)
    plt.figure()
    plt.imshow(detJacobian)
    print 'Range:', detJacobian.min(), detJacobian.max()
    X1,X0=np.mgrid[0:displacement_clean.shape[0], 0:displacement_clean.shape[1]]
    CS=plt.contour(X0,X1,detJacobian,levels=[0.0], colors='b')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('det(J(displacement))')
    #displacement=displacement_clean+np.random.normal(0.0, 0.0, displacement_clean.shape)
    displacement=displacement_clean
    #inverse=rcommon.invert_vector_field_fixed_point(displacement, 100, 1e-7)
    inverse=tf.invert_vector_field(displacement, 0.1, 100, 1e-7)
    residual=np.array(tf.compose_vector_fields(displacement_clean, inverse))
    [d,invd,res]=rcommon.plotDiffeomorphism(displacement, inverse, residual, 'invertible', 7)
    
#python registrationDiffeomorphic.py IBSR_01_ana_strip.nii.gz t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt 100

#testEstimateMultimodalDiffeomorphicField3DMultiScale('IBSR_01_ana_strip.nii.gz', 't1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 'IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt', 100)

#testEstimateMultimodalDiffeomorphicField3DMultiScale('IBSR_01_ana_strip_t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 't1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', None, 100)

#python registrationDiffeomorphic.py IBSR_01_ana_strip.nii.gz t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt 100
if __name__=='__main__':
    moving=sys.argv[1]
    fixed=sys.argv[2]
    affine=sys.argv[3]
    lambdaParam=np.float(sys.argv[4])
    #testEstimateMonomodalDiffeomorphicField3DMultiScale(0.1)
    testEstimateMultimodalDiffeomorphicField3DMultiScale(moving, fixed, affine, lambdaParam)
    #testInversion(5)
    #testInversion_invertible()
#    testCircleToCMonomodalDiffeomorphic(5)
    #######################################
#    maxOuterIter=[500,500,500,500,500,500]
#    runAllArcesExperiments(2000, maxOuterIter)
