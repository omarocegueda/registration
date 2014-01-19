import sys
import os
import numpy as np
import nibabel as nib
import registrationCommon as rcommon
import tensorFieldUtils as tf
from RegistrationOptimizer import RegistrationOptimizer
from EMMetric import EMMetric
import UpdateRule
from scipy import interp

def saveDeformedLattice3D(displacement, oname):
    minVal, maxVal=tf.get_displacement_range(displacement, None)
    sh=np.array([np.ceil(maxVal[0]),np.ceil(maxVal[1]),np.ceil(maxVal[2])], dtype=np.int32)
    L=np.array(rcommon.drawLattice3D(sh, 10))
    warped=np.array(tf.warp_volume(L, displacement, np.eye(4))).astype(np.int16)
    img=nib.Nifti1Image(warped, np.eye(4))
    img.to_filename(oname)

def histeq(im,nbr_bins=256):
  """  Histogram equalization of a grayscale image. """
  print 'Equalizing'
  # get image histogram
  imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
  cdf = imhist.cumsum() # cumulative distribution function
  cdf = 255 * cdf / cdf[-1] # normalize
  # use linear interpolation of cdf to find new pixel values
  im2 = interp(im.flatten(),bins[:-1],cdf)
  return im2.reshape(im.shape)

def registerMultimodalDiffeomorphic3D(fnameMoving, fnameFixed, fnameAffine, warpDir, lambdaParam):
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
    #print initAffine
    moving=moving.get_data().squeeze().astype(np.float64)
    fixed=fixed.get_data().squeeze().astype(np.float64)
    moving=histeq(moving)
    fixed=histeq(fixed)
    #moving=np.copy(moving, order='C')
    #fixed=np.copy(fixed, order='C')
    moving=moving.copy(order='C')
    fixed=fixed.copy(order='C')
    moving=(moving-moving.min())/(moving.max()-moving.min())
    fixed=(fixed-fixed.min())/(fixed.max()-fixed.min())
    maxOuterIter=[25,50,100,100]
    baseMoving=rcommon.getBaseFileName(fnameMoving)
    baseFixed=rcommon.getBaseFileName(fnameFixed)
    ###################Run registration##################
    similarityMetric=EMMetric({'symmetric':True, 
                               'lambda':lambdaParam, 
                               'stepType':EMMetric.GAUSS_SEIDEL_STEP, 
                               'qLevels':256,
                               'useDoubleGradient':True,
                               'maxInnerIter':10})
    updateRule=UpdateRule.Composition()
    registrationOptimizer=RegistrationOptimizer(fixed, moving, None, initAffine, similarityMetric, updateRule, maxOuterIter)
    registrationOptimizer.optimize()
    #####################################################
    displacement=registrationOptimizer.getForward()
    tf.prepend_affine_to_displacement_field(displacement, initAffine)
    #####Warp all requested volumes
    #---first the target using tri-linear interpolation---
    moving=nib.load(fnameMoving).get_data().squeeze().astype(np.float64)
    #moving=np.copy(moving, order='C')
    moving=moving.copy(order='C')
    warped=np.array(tf.warp_volume(moving, displacement)).astype(np.int16)
    imgWarped=nib.Nifti1Image(warped, F)
    imgWarped.to_filename('warpedDiff_'+baseMoving+'_'+baseFixed+'.nii.gz')
    #---warp using affine only
    moving=nib.load(fnameMoving).get_data().squeeze().astype(np.int32)
    #moving=np.copy(moving, order='C')
    moving=moving.copy(order='C')
    warped=np.array(tf.warp_discrete_volumeNNAffine(moving, referenceShape, initAffine)).astype(np.int16)
    imgWarped=nib.Nifti1Image(warped, F)#The affine transformation is the reference's one
    imgWarped.to_filename('warpedAffine_'+baseMoving+'_'+baseFixed+'.nii.gz')
    #---now the rest of the targets using nearest neighbor
    names=[os.path.join(warpDir,name) for name in os.listdir(warpDir)]
    for name in names:
        #---warp using the non-linear deformation
        toWarp=nib.load(name).get_data().squeeze().astype(np.int32)
        #toWarp=np.copy(toWarp, order='C')
        toWarp=toWarp.copy(order='C')
        baseWarp=rcommon.getBaseFileName(name)
        warped=np.array(tf.warp_discrete_volumeNN(toWarp, displacement)).astype(np.int16)
        imgWarped=nib.Nifti1Image(warped, F)#The affine transformation is the reference's one
        imgWarped.to_filename('warpedDiff_'+baseWarp+'_'+baseFixed+'.nii.gz')
        #---warp using affine only
#        warped=np.array(tf.warp_discrete_volumeNNAffine(toWarp, referenceShape, initAffine)).astype(np.int16)
#        imgWarped=nib.Nifti1Image(warped, F)#The affine transformation is the reference's one
#        imgWarped.to_filename('warpedAffine_'+baseWarp+'_'+baseFixed+'.nii.gz')
    #---finally, the deformed lattice
    saveDeformedLattice3D(displacement, 'latticeDispDiff_'+baseMoving+'_'+baseFixed+'.nii.gz')

'''
import dipyreg
dipyreg.registerMultimodalDiffeomorphic3D('target/IBSR_13_ana_strip.nii.gz', 'reference/IBSR_10_ana_strip.nii.gz', '../affine/IBSR_13_ana_strip_IBSR_10_ana_stripAffine.txt', 'warp', 50)
'''
if __name__=='__main__':
    '''
    python dipyreg.py "/opt/registration/data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz" "/opt/registration/data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz" "IBSR_01_ana_strip_IBSR_02_ana_stripAffine.txt" "warp" 100.0
    '''
    print sys.version
    moving=sys.argv[1]
    fixed=sys.argv[2]
    affine=sys.argv[3]
    warpDir=sys.argv[4]
    lambdaParam=np.float(sys.argv[5])
    registerMultimodalDiffeomorphic3D(moving, fixed, affine, warpDir, lambdaParam)
