"""
This script is the main launcher of the multi-modal non-linear image registration
"""
import sys
import os
import numpy as np
import nibabel as nib
import registrationCommon as rcommon
import tensorFieldUtils as tf
from SymmetricRegistrationOptimizer import SymmetricRegistrationOptimizer
from EMMetric import EMMetric
import UpdateRule
from dipy.fixes import argparse as arg

parser = arg.ArgumentParser(description='Multi-modal, non-linear image registration')

parser.add_argument('target', action='store', metavar='target',
                    help='Nifti1 image (*.nii or *.nii.gz) or other formats supported by Nibabel')

parser.add_argument('reference', action='store', metavar='reference',
                    help='Nifti1 image (*.nii or *.nii.gz) or other formats supported by Nibabel')

parser.add_argument('affine', action='store', metavar='affine',
                    help='ANTS affine registration matrix (.txt) that registers target to reference')

parser.add_argument('warp_dir', action='store', metavar='warp_dir',
                    help='Directory (relative to ./ ) containing the images to be warped with the obtained deformation field')

parser.add_argument('--smooth', action='store', metavar='smooth',
                    help='A scalar to be used as regularization parameter (higher values produce smoother deformation fields)')

parser.add_argument('--iter', action='store', metavar='max_iter',
                    help='A x-separated list of integers indicating the maximum number of iterations at each level of the Gaussian Pyramid (similar to ANTS), e.g. 10x100x100')

params = parser.parse_args()


def saveDeformedLattice3D(displacement, oname):
    minVal, maxVal=tf.get_displacement_range(displacement, None)
    sh=np.array([np.ceil(maxVal[0]),np.ceil(maxVal[1]),np.ceil(maxVal[2])], dtype=np.int32)
    L=np.array(rcommon.drawLattice3D(sh, 10))
    warped=np.array(tf.warp_volume(L, displacement)).astype(np.int16)
    img=nib.Nifti1Image(warped, np.eye(4))
    img.to_filename(oname)

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
    moving=moving.copy(order='C')
    fixed=fixed.copy(order='C')
    moving=(moving-moving.min())/(moving.max()-moving.min())
    fixed=(fixed-fixed.min())/(fixed.max()-fixed.min())
    baseMoving=rcommon.getBaseFileName(fnameMoving)
    baseFixed=rcommon.getBaseFileName(fnameFixed)
    ###################Run registration##################
    metricParameters={'symmetric':True, 
                      'lambda':lambdaParam, 
                      'stepType':EMMetric.GAUSS_SEIDEL_STEP, 
                      'qLevels':256,
                      'useDoubleGradient':True,
                      'maxInnerIter':20,
                      'iterationType':'vCycle'}
    similarityMetric=EMMetric(metricParameters)
    updateRule=UpdateRule.Composition()
    optimizerParameters={'maxIter':[25,50,100], 'inversionIter':20,
                'inversionTolerance':1e-3, 'tolerance':1e-6, 
                'reportStatus':True}
    registrationOptimizer=SymmetricRegistrationOptimizer(fixed, moving, None, initAffine, similarityMetric, updateRule, optimizerParameters)
    registrationOptimizer.optimize()
    #####################################################
    displacement=registrationOptimizer.getForward()
    del registrationOptimizer
    del similarityMetric
    del updateRule
    #####Warp all requested volumes
    #---first the target using tri-linear interpolation---
    moving=nib.load(fnameMoving).get_data().squeeze().astype(np.float64)
    moving=moving.copy(order='C')
    warped=np.array(tf.warp_volume(moving, displacement)).astype(np.int16)
    imgWarped=nib.Nifti1Image(warped, F)
    imgWarped.to_filename('warpedDiff_'+baseMoving+'_'+baseFixed+'.nii.gz')
    #---warp using affine only
    moving=nib.load(fnameMoving).get_data().squeeze().astype(np.int32)
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
    #---finally, the deformed lattice
    saveDeformedLattice3D(displacement, 'latticeDispDiff_'+baseMoving+'_'+baseFixed+'.nii.gz')
'''
import dipyreg
dipyreg.registerMultimodalDiffeomorphic3D('target/IBSR_16_ana_strip.nii.gz', 'reference/IBSR_10_ana_strip.nii.gz', '../affine/IBSR_16_ana_strip_IBSR_10_ana_stripAffine.txt', 'warp', 25)
dipyreg.registerMultimodalDiffeomorphic3D('target/IBSR_07_ana_strip.nii.gz', 'reference/IBSR_17_ana_strip.nii.gz', '../affine/IBSR_07_ana_strip_IBSR_17_ana_stripAffine.txt', 'warp', 50)
dipyreg.registerMultimodalDiffeomorphic3D('target/IBSR_13_ana_strip.nii.gz', 'reference/IBSR_10_ana_strip.nii.gz', '../affine/IBSR_13_ana_strip_IBSR_10_ana_stripAffine.txt', 'warp', 50)
'''
if __name__=='__main__':
    '''
    python dipyreg.py "/opt/registration/data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz" "/opt/registration/data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz" "IBSR_01_ana_strip_IBSR_02_ana_stripAffine.txt" "warp" 100.0
    '''
#    print sys.version
#    moving=sys.argv[1]
#    fixed=sys.argv[2]
#    affine=sys.argv[3]
#    warpDir=sys.argv[4]
#    lambdaParam=np.float(sys.argv[5])
    print params.target, params.reference, params.affine, params.warp_dir, float(params.smooth)
    registerMultimodalDiffeomorphic3D(params.target, params.reference, params.affine, params.warp_dir, float(params.smooth))
