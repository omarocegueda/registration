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

parser = arg.ArgumentParser(description=r'''Multi-modal, non-linear image registration. By default, it does NOT save the
                                        resulting deformation field but only the deformed target image using tri-linear
                                        interpolation under two transformations: (1)the affine transformation used to pre-register
                                        the images and (2) the resulting displacement field. If a warp directory is specified,
                                        then each image inside the directory will be warped using nearest-neighbor interpolation
                                        under the resulting deformation field. 
                                        
                                        The name of the warped target image under the AFFINE
                                        transformation will be 'warpedAffine_'+baseTarget+'_'+baseReference+'.nii.gz', where baseTarget 
                                        is the file name of the target image (excluding all characters at and after the first '.'), 
                                        baseReference is the file name of the reference image (excluding all characters at and after 
                                        the first '.'). For example, if target is 'a.nii.gz' and reference is 'b.nii.gz', then the 
                                        resulting deformation field will be saved as 'dispDiff_a_b.npy'.
                                        
                                        The name of the warped target image under the NON-LINEAR
                                        transformation will be 'warpedAffine_'+baseTarget+'_'+baseReference+'.nii.gz', with the convention
                                        explained above.
                                        
                                        Similarly, the name of each warped image inside the warp directory will be
                                        'warpedDiff_'+baseWarp+'_'+baseReference+'.nii.gz' where baseWarp is the 
                                        name of the corresponding image following the same convetion as above.
                                        
                                        Example:
                                        
                                        python dipyreg.py target.nii.gz reference.nii.gz ANTS_affine_target_reference.txt --smooth=25.0 --iter 10,50,50''')

parser.add_argument('target', action='store', metavar='target',
                    help='Nifti1 image (*.nii or *.nii.gz) or other formats supported by Nibabel')

parser.add_argument('reference', action='store', metavar='reference',
                    help='Nifti1 image (*.nii or *.nii.gz) or other formats supported by Nibabel')

parser.add_argument('affine', action='store', metavar='affine',
                    help='ANTS affine registration matrix (.txt) that registers target to reference')

parser.add_argument('warp_dir', action='store', metavar='warp_dir',
                    help='Directory (relative to ./ ) containing the images to be warped with the obtained deformation field')

parser.add_argument('-s', '--smooth', action='store', metavar='lambda',
                    help='The regulaization parameter (lambda) of the deformation field (higher values produce smoother deformation fields)',
                    default='25.0')

parser.add_argument('-i', '--iter', action='store', metavar='i_0,i_1,...,i_n',
                    help='A comma-separated list of integers indicating the maximum number of iterations at each level of the Gaussian Pyramid (similar to ANTS), e.g. 10,100,100',
                    default='25,50,100')

parser.add_argument('-inv_iter', '--inversion_iter', action='store', metavar='max_iter',
                    help='The maximum number of iterations for the displacement field inversion algorithm',
                    default='20')

parser.add_argument('-inv_tol', '--inversion_tolerance', action='store', metavar='tolerance',
                    help='The tolerance for the displacement field inversion algorithm',
                    default='1e-3')

parser.add_argument('-ii', '--inner_iter', action='store', metavar='max_iter',
                    help='The number of Gauss-Seidel iterations to be performed to minimize each linearized energy',
                    default='20')

parser.add_argument('-ql', '--quantization_levels', action='store', metavar='qLevels',
                    help='The number levels to be used for the EM quantization',
                    default='256')

parser.add_argument('-single', '--single_gradient', action='store_true',
                    help='The number levels to be used for the EM quantization')

parser.add_argument('-rs', '--report_status', action='store_true',
                    help='Instructs the algorithm to show the overlaid registered images after each pyramid level')

parser.add_argument('-sd', '--save_displacement', dest='output_list', action='append_const', const='displacement',
                    help=r'''Specifies that the displacement field must be saved. The displacement field will be 
                    saved in .npy format. The filename will be the concatenation:
                    'dispDiff_'+baseTarget+'_'+baseReference+'.npy' where baseTarget is the file name of the
                    target image (excluding all characters at and after the first '.'), baseReference is the file 
                    name of the reference image (excluding all characters at and after the first '.'). For example,
                    if target is 'a.nii.gz' and reference is 'b.nii.gz', then the resulting deformation field will
                    be saved as 'dispDiff_a_b.npy'.''')

parser.add_argument('-si', '--save_inverse', dest='output_list', action='append_const', const='inverse',
                    help=r'''Specifies that the inverse displacement field must be saved. The displacement field will be 
                    saved in .npy format. The filename will be the concatenation:
                    'invDispDiff_'+baseTarget+'_'+baseReference+'.npy' where baseTarget is the file name of the
                    target image (excluding all characters at and after the first '.'), baseReference is the file 
                    name of the reference image (excluding all characters at and after the first '.'). For example,
                    if target is 'a.nii.gz' and reference is 'b.nii.gz', then the resulting deformation field will
                    be saved as 'invDispDiff_a_b.npy'.''')

parser.add_argument('-sl', '--save_lattice', dest='output_list', action='append_const', const='lattice',
                    help=r'''Specifies that the deformation lattice (i.e., the obtained deformation field 
                    applied to a regular lattice) must be saved. The displacement field will be 
                    saved in .npy format. The filename will be the concatenation:
                    'latticeDispDiff_'+baseTarget+'_'+baseReference+'.npy' where baseTarget is the file name of the
                    target image (excluding all characters at and after the first '.'), baseReference is the file 
                    name of the reference image (excluding all characters at and after the first '.'). For example,
                    if target is 'a.nii.gz' and reference is 'b.nii.gz', then the resulting deformation field will
                    be saved as 'latticeDispDiff_a_b.npy'.''')
                    
parser.add_argument('-sil', '--save_inverse_lattice', dest='output_list', action='append_const', const='inv_lattice',
                    help=r'''Specifies that the inverse deformation lattice (i.e., the obtained inverse deformation 
                    field applied to a regular lattice) must be saved. The displacement field will be 
                    saved in .npy format. The filename will be the concatenation:
                    'invLatticeDispDiff_'+baseTarget+'_'+baseReference+'.npy' where baseTarget is the file name of the
                    target image (excluding all characters at and after the first '.'), baseReference is the file 
                    name of the reference image (excluding all characters at and after the first '.'). For example,
                    if target is 'a.nii.gz' and reference is 'b.nii.gz', then the resulting deformation field will
                    be saved as 'invLatticeDispDiff_a_b.npy'.''')

def checkArguments(params):
    print params.target, params.reference, params.affine, params.warp_dir, float(params.smooth)
    print 'iter:',[int(i) for i in params.iter.split('x')]
    print params.inner_iter, params.quantization_levels, params.single_gradient
    print 'Inversion:',params.inversion_iter, params.inversion_tolerance
    print '---------Output requested--------------'
    print params.output_list

def saveDeformedLattice3D(displacement, oname):
    minVal, maxVal=tf.get_displacement_range(displacement, None)
    sh=np.array([np.ceil(maxVal[0]),np.ceil(maxVal[1]),np.ceil(maxVal[2])], dtype=np.int32)
    L=np.array(rcommon.drawLattice3D(sh, 10))
    warped=np.array(tf.warp_volume(L, displacement)).astype(np.int16)
    img=nib.Nifti1Image(warped, np.eye(4))
    img.to_filename(oname)

def registerMultimodalDiffeomorphic3D(params):
    fnameMoving=params.target
    fnameFixed=params.reference
    fnameAffine=params.affine
    warpDir=params.warp_dir
    print 'Registering', fnameMoving, 'to', fnameFixed
    sys.stdout.flush()
    lambdaParam=float(params.smooth)
    qLevels=int(params.quantization_levels)
    useDoubleGradient=False if params.single_gradient else True
    innerIter=int(params.inner_iter)
    maxIter=[int(i) for i in params.iter.split('x')]
    inversionIter=int(params.inversion_iter)
    inversionTolerance=float(params.inversion_tolerance)
    reportStatus=True if params.report_status else False
    ####Initialize parameter dictionaries####
    metricParameters={'lambda':lambdaParam,
                      'qLevels':qLevels,
                      'useDoubleGradient':useDoubleGradient,
                      'maxInnerIter':innerIter}
    optimizerParameters={'maxIter':maxIter, 
                         'inversionIter':inversionIter,
                         'inversionTolerance':inversionTolerance,
                         'reportStatus':reportStatus}
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
    ###################Run registration##################
    similarityMetric=EMMetric(metricParameters)
    updateRule=UpdateRule.Composition()
    registrationOptimizer=SymmetricRegistrationOptimizer(fixed, moving, None, initAffine, similarityMetric, updateRule, optimizerParameters)
    registrationOptimizer.optimize()
    displacement=registrationOptimizer.getForward()
    inverse=registrationOptimizer.getBackward()
    del registrationOptimizer
    del similarityMetric
    del updateRule
    #---warp the target image using the obtained deformation field---
    baseMoving=rcommon.getBaseFileName(fnameMoving)
    baseFixed=rcommon.getBaseFileName(fnameFixed)
    moving=nib.load(fnameMoving).get_data().squeeze().astype(np.float64)
    moving=moving.copy(order='C')
    warped=np.array(tf.warp_volume(moving, displacement)).astype(np.int16)
    imgWarped=nib.Nifti1Image(warped, F)
    imgWarped.to_filename('warpedDiff_'+baseMoving+'_'+baseFixed+'.nii.gz')
    #---warp the target image using the affine transformation only---
    moving=nib.load(fnameMoving).get_data().squeeze().astype(np.float64)
    moving=moving.copy(order='C')
    warped=np.array(tf.warp_volume_affine(moving, referenceShape, initAffine)).astype(np.int16)
    imgWarped=nib.Nifti1Image(warped, F)#The affine transformation is the reference's one
    imgWarped.to_filename('warpedAffine_'+baseMoving+'_'+baseFixed+'.nii.gz')
    #---warp all volums in the warp directory using Nearest Neighbor interpolation
    names=[os.path.join(warpDir,name) for name in os.listdir(warpDir)]
    for name in names:
        toWarp=nib.load(name).get_data().squeeze().astype(np.int32)
        toWarp=toWarp.copy(order='C')
        baseWarp=rcommon.getBaseFileName(name)
        warped=np.array(tf.warp_discrete_volumeNN(toWarp, displacement)).astype(np.int16)
        imgWarped=nib.Nifti1Image(warped, F)#The affine transformation is the reference's one
        imgWarped.to_filename('warpedDiff_'+baseWarp+'_'+baseFixed+'.nii.gz')
    #---finally, the optional output
    if params.output_list==None:
        return
    if 'lattice' in params.output_list:
        saveDeformedLattice3D(displacement, 'latticeDispDiff_'+baseMoving+'_'+baseFixed+'.nii.gz')
    if 'inv_lattice' in params.output_list:
        saveDeformedLattice3D(inverse, 'invLatticeDispDiff_'+baseMoving+'_'+baseFixed+'.nii.gz')
    if 'displacement' in params.output_list:
        np.save('dispDiff_'+baseMoving+'_'+baseFixed+'.npy', displacement)
    if 'inverse' in params.output_list:
        np.save('invDispDiff_'+baseMoving+'_'+baseFixed+'.npy', inverse)
if __name__=='__main__':
    registerMultimodalDiffeomorphic3D(parser.parse_args())
