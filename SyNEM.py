import numpy as np
import scipy as sp
import tensorFieldUtils as tf
import nibabel as nib
import matplotlib.pyplot as plt
import registrationCommon as rcommon
import os
import sys
###############################################################
####### Symmetric Monomodal registration - EM (2D)############
###############################################################
def estimateNewMonomodalSyNField2D(moving, fixed, fWarp, fInv, mWarp, mInv, lambdaParam, maxOuterIter):
    '''
    Warning: in the monomodal case, the parameter lambda must be significantly lower than in the multimodal case. Try lambdaParam=1,
    as opposed as lambdaParam=150 used in the multimodal case
    '''
    innerTolerance=1e-4
    outerTolerance=1e-3
    
    if(mWarp!=None):
        totalM=mWarp
        totalMInv=mInv
    else:
        totalM=np.zeros(shape=(fixed.shape)+(2,), dtype=np.float64)
        totalMInv=np.zeros(shape=(fixed.shape)+(2,), dtype=np.float64)
    if(fWarp!=None):
        totalF=fWarp
        totalFInv=fInv
    else:
        totalF=np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
        totalFInv=np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
    outerIter=0
    framesToCapture=5
    maxOuterIter=framesToCapture*((maxOuterIter+framesToCapture-1)/framesToCapture)
    itersPerCapture=maxOuterIter/framesToCapture
    plt.figure()
    while(outerIter<maxOuterIter):
        outerIter+=1
        print 'Outer iter:', outerIter
        wmoving=np.array(tf.warp_image(moving, totalMInv))
        wfixed=np.array(tf.warp_image(fixed, totalFInv))
        if((outerIter==1) or (outerIter%itersPerCapture==0)):
            plt.subplot(1,framesToCapture+1, 1+outerIter/itersPerCapture)
            rcommon.overlayImages(wmoving, wfixed, False)
            plt.title('Iter:'+str(outerIter-1))
        #Compute forward update
        sigmaField=np.ones_like(wmoving, dtype=np.float64)
        deltaField=wfixed-wmoving
        movingGradient    =np.empty(shape=(wmoving.shape)+(2,), dtype=np.float64)
        movingGradient[:,:,0], movingGradient[:,:,1]=sp.gradient(wmoving)
        maxVariation=1+innerTolerance
        innerIter=0
        fw     =np.zeros(shape=(fixed.shape)+(2,), dtype=np.float64)
        maxInnerIter=1000
        while((maxVariation>innerTolerance)and(innerIter<maxInnerIter)):
            innerIter+=1
            maxVariation=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, movingGradient,  lambdaParam, fw, None)
        #fw*=0.5
        totalF, stats=tf.compose_vector_fields(fw, totalF)
        totalF=np.array(totalF);
        meanDispF=np.mean(np.abs(fw))
        #Compute backward field
        sigmaField=np.ones_like(wfixed, dtype=np.float64)
        deltaField=wmoving-wfixed
        fixedGradient    =np.empty(shape=(wfixed.shape)+(2,), dtype=np.float64)
        fixedGradient[:,:,0], fixedGradient[:,:,1]=sp.gradient(wfixed)
        maxVariation=1+innerTolerance
        innerIter=0
        mw     =np.zeros(shape=(fixed.shape)+(2,), dtype=np.float64)
        maxInnerIter=1000
        while((maxVariation>innerTolerance)and(innerIter<maxInnerIter)):
            innerIter+=1
            maxVariation=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, fixedGradient,  lambdaParam, mw, None)
        #mw*=0.5
        totalM, stats=tf.compose_vector_fields(mw, totalM)
        totalM=np.array(totalM);
        meanDispM=np.mean(np.abs(mw))
        totalFInv=np.array(tf.invert_vector_field_fixed_point(totalF, None, 20, 1e-3, None))
        totalMInv=np.array(tf.invert_vector_field_fixed_point(totalM, None, 20, 1e-3, None))
        totalF=np.array(tf.invert_vector_field_fixed_point(totalFInv, None, 20, 1e-3, None))
        totalM=np.array(tf.invert_vector_field_fixed_point(totalMInv, None, 20, 1e-3, None))
#        totalFInv=np.array(tf.invert_vector_field(totalF, 0.75, 100, 1e-6))
#        totalMInv=np.array(tf.invert_vector_field(totalM, 0.75, 100, 1e-6))
#        totalF=np.array(tf.invert_vector_field(totalFInv, 0.75, 100, 1e-6))
#        totalM=np.array(tf.invert_vector_field(totalMInv, 0.75, 100, 1e-6))
        if(meanDispM+meanDispF<2*outerTolerance):
            break
    print "Iter: ",innerIter, "Mean lateral displacement:", 0.5*(meanDispM+meanDispF), "Max variation:",maxVariation
    return totalF, totalFInv, totalM, totalMInv

def estimateMonomodalSyNField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level, displacementList):
    n=len(movingPyramid)
    if(level==(n-1)):
        totalF, totalFInv, totalM, totalMInv=estimateNewMonomodalSyNField2D(movingPyramid[level], fixedPyramid[level], None, None, None, None, lambdaParam, maxOuterIter[level])
        if(displacementList!=None):
            displacementList.insert(0,totalM)
        return totalF, totalFInv, totalM, totalMInv
    subF, subFInv, subM, subMInv=estimateMonomodalSyNField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level+1, displacementList)
    sh=np.array(fixedPyramid[level].shape).astype(np.int32)
    upF=np.array(tf.upsample_displacement_field(subF, sh))*2
    upFInv=np.array(tf.upsample_displacement_field(subFInv, sh))*2
    upM=np.array(tf.upsample_displacement_field(subM, sh))*2
    upMInv=np.array(tf.upsample_displacement_field(subMInv, sh))*2
    totalF, totalFInv, totalM, totalMInv=estimateNewMonomodalSyNField2D(movingPyramid[level], fixedPyramid[level], upF, upFInv, upM, upMInv, lambdaParam, maxOuterIter[level])
    if(displacementList!=None):
        displacementList.insert(0, totalM)
    if(level==0):
        totalF=np.array(tf.compose_vector_fields(totalF, totalMInv))
        totalM=np.array(tf.compose_vector_fields(totalM, totalFInv))
        return totalM, totalF
    return totalF, totalFInv, totalM, totalMInv

def testEstimateMonomodalSyNField2DMultiScale(lambdaParam):
    fname0='IBSR_01_to_02.nii.gz'
    fname1='data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz'
    nib_moving = nib.load(fname0)
    nib_fixed= nib.load(fname1)
    moving=nib_moving.get_data().squeeze()
    fixed=nib_fixed.get_data().squeeze()
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
    displacement=estimateMonomodalSyNField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxIter, 0,displacementList)
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

def testCircleToCMonomodalSyNEM(lambdaParam, maxOuterIter):
    fname0='data/circle.png'
    #fname0='data/C_trans.png'
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
    movingPyramid=[img for img in rcommon.pyramid_gaussian_2D(moving, level, maskMoving)]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_2D(fixed, level, maskFixed)]
    rcommon.plotOverlaidPyramids(movingPyramid, fixedPyramid)
    displacementList=[]
    displacement, dinv=estimateMonomodalSyNField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0,displacementList)
    inverse=np.array(tf.invert_vector_field(displacement, 0.75, 300, 1e-7))
    residual, stats=tf.compose_vector_fields(displacement, inverse)
    residual=np.array(residual)
    warpPyramid=[rcommon.warpImage(movingPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids(warpPyramid, fixedPyramid)
    rcommon.overlayImages(warpPyramid[0], fixedPyramid[0])
    rcommon.plotDiffeomorphism(displacement, inverse, residual, '',7)

def estimateNewMultimodalSyNField3D(moving, fixed, fWarp, fInv, mWarp, mInv, initAffine, lambdaDisplacement, quantizationLevels, maxOuterIter, reportProgress=False):
    '''
        fwWarp: forward warp, the displacement field that warps moving towards fixed
        bwWarp: backward warp, the displacement field that warps fixed towards moving
        initAffine: the affine transformation to bring moving over fixed (this part is not symmetric)
    '''
    print 'Moving shape:',moving.shape,'. Fixed shape:',fixed.shape
    innerTolerance=1e-3
    outerTolerance=1e-3
    fixedMask=(fixed>0).astype(np.int32)
    movingMask=(moving>0).astype(np.int32)
    if(fWarp!=None):
        totalF=fWarp
        totalFInv=fInv
    else:
        totalF    =np.zeros(shape=(fixed.shape)+(3,), dtype=np.float64)
        totalFInv =np.zeros(shape=(fixed.shape)+(3,), dtype=np.float64)
    if(mWarp!=None):
        totalM=mWarp
        totalMInv=mInv
    else:
        totalM   =np.zeros(shape=(moving.shape)+(3,), dtype=np.float64)
        totalMInv=np.zeros(shape=(moving.shape)+(3,), dtype=np.float64)
    finished=False
    outerIter=0
    while((not finished) and (outerIter<maxOuterIter)):
        outerIter+=1
        if(reportProgress):
            print 'Iter:',outerIter,'/',maxOuterIter
        #---E step---
        wmoving=np.array(tf.warp_volume(moving, totalMInv, initAffine))
        wmovingMask=np.array(tf.warp_discrete_volumeNN(movingMask, totalMInv, initAffine)).astype(np.int32)
        wfixed=np.array(tf.warp_volume(fixed, totalFInv))
        wfixedMask=np.array(tf.warp_discrete_volumeNN(fixedMask, totalFInv)).astype(np.int32)
        fixedQ, grayLevels, hist=tf.quantizePositiveVolumeCYTHON(wfixed, quantizationLevels)
        fixedQ=np.array(fixedQ, dtype=np.int32)
        movingQ, grayLevels, hist=tf.quantizePositiveVolumeCYTHON(wmoving, quantizationLevels)
        movingQ=np.array(movingQ, dtype=np.int32)
        trust=wfixedMask*wmovingMask
        meansMoving, variancesMoving=tf.computeMaskedVolumeClassStatsCYTHON(trust, wmoving, quantizationLevels, fixedQ)
        meansFixed, variancesFixed=tf.computeMaskedVolumeClassStatsCYTHON(trust, wfixed, quantizationLevels, movingQ)
        meansMoving[0]=0
        meansFixed[0]=0
        meansMoving=np.array(meansMoving)
        meansFixed=np.array(meansFixed)
        variancesMoving=np.array(variancesMoving)
        sigmaFieldMoving=variancesMoving[fixedQ]
        variancesFixed=np.array(variancesFixed)
        sigmaFieldFixed=variancesFixed[movingQ]
        deltaFieldMoving=meansMoving[fixedQ]-wmoving
        deltaFieldFixed=meansFixed[movingQ]-wfixed
        #--M step--
        movingGradient  =np.empty(shape=(moving.shape)+(3,), dtype=np.float64)
        movingGradient[:,:,:,0], movingGradient[:,:,:,1], movingGradient[:,:,:,2]=sp.gradient(wmoving)
        #iterate forward field
        maxVariation=1+innerTolerance
        innerIter=0
        maxInnerIter=100
        fw=np.zeros_like(totalF)
        while((maxVariation>innerTolerance)and(innerIter<maxInnerIter)):
            innerIter+=1
            maxVariation=tf.iterateDisplacementField3DCYTHON(deltaFieldMoving, sigmaFieldMoving, movingGradient,  lambdaDisplacement, totalF, fw, None)
        del movingGradient
        fw*=0.5
        totalF=np.array(tf.compose_vector_fields3D(fw, totalF))#Multiply fw by 0.5??
        nrm=np.sqrt(fw[...,0]**2+fw[...,1]**2+fw[...,2]**2)
        del fw        
        #iterate backward field
        fixedGradient   =np.empty(shape=(fixed.shape)+(3,), dtype=np.float64)
        fixedGradient[:,:,:,0], fixedGradient[:,:,:,1], fixedGradient[:,:,:,2]=sp.gradient(wfixed)
        maxVariation=1+innerTolerance
        innerIter=0
        maxInnerIter=100
        mw=np.zeros_like(totalM)
        while((maxVariation>innerTolerance)and(innerIter<maxInnerIter)):
            innerIter+=1
            maxVariation=tf.iterateDisplacementField3DCYTHON(deltaFieldFixed, sigmaFieldFixed, fixedGradient,  lambdaDisplacement, totalM, mw, None)
        del fixedGradient
        mw*=0.5
        totalM=np.array(tf.compose_vector_fields3D(mw, totalM))#Multiply bw by 0.5??
        nrm=np.sqrt(mw[...,0]**2+mw[...,1]**2+mw[...,2]**2)
        del mw        
        #invert fields
        totalFInv=np.array(tf.invert_vector_field_fixed_point3D(totalF, 20, 1e-6))
        totalMInv=np.array(tf.invert_vector_field_fixed_point3D(totalM, 20, 1e-6))
        totalF=np.array(tf.invert_vector_field_fixed_point3D(totalFInv, 20, 1e-6))
        totalM=np.array(tf.invert_vector_field_fixed_point3D(totalMInv, 20, 1e-6))
        maxDisplacement=np.mean(nrm)
        if((maxDisplacement<outerTolerance)or(outerIter>=maxOuterIter)):
            finished=True
    print "Iter: ",outerIter, "Mean displacement:", maxDisplacement, "Max variation:",maxVariation
    return totalF, totalFInv, totalM, totalMInv

def estimateMultimodalSyN3DMultiScale(movingPyramid, fixedPyramid, initAffine, lambdaParam, maxOuterIter, level=0):
    n=len(movingPyramid)
    quantizationLevels=256
    if(level==(n-1)):
        totalF, totalFInv, totalM, totalMInv=estimateNewMultimodalSyNField3D(movingPyramid[level], fixedPyramid[level], None, None, None, None, initAffine, lambdaParam, quantizationLevels, maxOuterIter[level], level==0)
        return totalF, totalFInv, totalM, totalMInv
    subAffine=initAffine.copy()
    subAffine[:3,3]*=0.5
    subF, subFInv, subM, subMInv=estimateMultimodalSyN3DMultiScale(movingPyramid, fixedPyramid, subAffine, lambdaParam, maxOuterIter, level+1)
    sh=np.array(fixedPyramid[level].shape).astype(np.int32)
    upF=np.array(tf.upsample_displacement_field3D(subF, sh))*2
    upFInv=np.array(tf.upsample_displacement_field3D(subFInv, sh))*2
    upM=np.array(tf.upsample_displacement_field3D(subM, sh))*2
    upMInv=np.array(tf.upsample_displacement_field3D(subMInv, sh))*2
    del subF
    del subFInv
    del subM
    del subMInv
    totalF, totalFInv, totalM, totalMInv=estimateNewMultimodalSyNField3D(movingPyramid[level], fixedPyramid[level], upF, upFInv, upM, upMInv, initAffine, lambdaParam, quantizationLevels, maxOuterIter[level], level==0)
    if level==0:
        totalF=np.array(tf.compose_vector_fields3D(totalF, totalMInv))#Multiply bw by 0.5??
        totalM=np.array(tf.compose_vector_fields3D(totalM, totalFInv))#Multiply bw by 0.5??
        return totalM, totalF
    return totalF, totalFInv, totalM, totalMInv

def saveDeformedLattice3D(displacement, oname):
    minVal, maxVal=tf.get_displacement_range(displacement, None)
    sh=np.array([np.ceil(maxVal[0]),np.ceil(maxVal[1]),np.ceil(maxVal[2])], dtype=np.int32)
    L=np.array(rcommon.drawLattice3D(sh, 10))
    warped=np.array(tf.warp_volume(L, displacement, np.eye(4))).astype(np.int16)
    img=nib.Nifti1Image(warped, np.eye(4))
    img.to_filename(oname)

def testEstimateMultimodalSyN3DMultiScale(fnameMoving, fnameFixed, fnameAffine, warpDir, lambdaParam):
    '''
        testEstimateMultimodalDiffeomorphicField3DMultiScale('IBSR_01_ana_strip.nii.gz', 't1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 'IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt', 100)
    '''
    print 'Registering', fnameMoving, 'to', fnameFixed,'with lambda=',lambdaParam  
    sys.stdout.flush()
    moving = nib.load(fnameMoving)
    fixed= nib.load(fnameFixed)
    referenceShape=np.array(fixed.shape, dtype=np.int32)
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
    level=2
    maskMoving=moving>0
    maskFixed=fixed>0
    movingPyramid=[img for img in rcommon.pyramid_gaussian_3D(moving, level, maskMoving)]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_3D(fixed, level, maskFixed)]
    #maxOuterIter=[25,50,100,100, 100, 100]
    maxOuterIter=[2,2,2,2,2,2]
    baseMoving=rcommon.getBaseFileName(fnameMoving)
    baseFixed=rcommon.getBaseFileName(fnameFixed)    
#    if(os.path.exists('disp_'+baseMoving+'_'+baseFixed+'.npy')):
#        displacement=np.load('disp_'+baseMoving+'_'+baseFixed+'.npy')
#    else:
    displacement, directInverse=estimateMultimodalSyN3DMultiScale(movingPyramid, fixedPyramid, initAffine, lambdaParam, maxOuterIter, 0)
    tf.prepend_affine_to_displacement_field(displacement, initAffine)
#    np.save('disp_'+baseMoving+'_'+baseFixed+'.npy', displacement)
    #####Warp all requested volumes
    #---first the target using tri-linear interpolation---
    moving=nib.load(fnameMoving).get_data().squeeze().astype(np.float64)
    moving=np.copy(moving, order='C')
    warped=np.array(tf.warp_volume(moving, displacement)).astype(np.int16)
    imgWarped=nib.Nifti1Image(warped, F)
    imgWarped.to_filename('warpedDiff_'+baseMoving+'_'+baseFixed+'.nii.gz')
    #---warp using affine only
    moving=nib.load(fnameMoving).get_data().squeeze().astype(np.int32)
    moving=np.copy(moving, order='C')
    warped=np.array(tf.warp_discrete_volumeNNAffine(moving, referenceShape, initAffine)).astype(np.int16)
    imgWarped=nib.Nifti1Image(warped, F)#The affine transformation is the reference's one
    imgWarped.to_filename('warpedAffine_'+baseMoving+'_'+baseFixed+'.nii.gz')
    #---now the rest of the targets using nearest neighbor
    names=[os.path.join(warpDir,name) for name in os.listdir(warpDir)]
    for name in names:
        #---warp using the non-linear deformation
        toWarp=nib.load(name).get_data().squeeze().astype(np.int32)
        toWarp=np.copy(toWarp, order='C')
        baseWarp=rcommon.getBaseFileName(name)
        warped=np.array(tf.warp_discrete_volumeNN(toWarp, displacement)).astype(np.int16)
        imgWarped=nib.Nifti1Image(warped, F)#The affine transformation is the reference's one
        imgWarped.to_filename('warpedDiff_'+baseWarp+'_'+baseFixed+'.nii.gz')
        #---warp using affine inly
        warped=np.array(tf.warp_discrete_volumeNNAffine(toWarp, referenceShape, initAffine)).astype(np.int16)
        imgWarped=nib.Nifti1Image(warped, F)#The affine transformation is the reference's one
        imgWarped.to_filename('warpedAffine_'+baseWarp+'_'+baseFixed+'.nii.gz')
    #---finally, the deformed lattices (forward, inverse and resdidual)---    
    lambdaParam=0.9
    maxIter=100
    tolerance=1e-4
    print 'Computing inverse...'
    inverse=np.array(tf.invert_vector_field3D(displacement, lambdaParam, maxIter, tolerance))
    residual=np.array(tf.compose_vector_fields3D(displacement, inverse))
    saveDeformedLattice3D(displacement, 'latticeDispDiff_'+baseMoving+'_'+baseFixed+'.nii.gz')
    saveDeformedLattice3D(inverse, 'latticeInvDiff_'+baseMoving+'_'+baseFixed+'.nii.gz')
    saveDeformedLattice3D(residual, 'latticeResdiff_'+baseMoving+'_'+baseFixed+'.nii.gz')
    residual=np.sqrt(np.sum(residual**2,3))
    print "Mean residual norm:", residual.mean()," (",residual.std(), "). Max residual norm:", residual.max()

if __name__=='__main__':
    '''
    python SyNEM.py "/opt/registration/data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz" "/opt/registration/data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz" "IBSR_01_ana_strip_IBSR_02_ana_stripAffine.txt" "warp" 100.0
    '''
    moving=sys.argv[1]
    fixed=sys.argv[2]
    affine=sys.argv[3]
    warpDir=sys.argv[4]
    lambdaParam=np.float(sys.argv[5])
    testEstimateMultimodalSyN3DMultiScale(moving, fixed, affine, warpDir, lambdaParam)
    #testCircleToCMonomodalSyNEM(5,[100,100,100,100])
    #testCircleToCMonomodalSyNEM(5,[50,50,50,50])