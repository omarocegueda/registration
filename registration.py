__author__ = 'khayyam'
import os
import numpy as np
import registrationRigid as rrigid
if __name__=='__main__':
    #the following command will compile the 'tensorFieldUtils' cython extension
    #the working directory must be properly set: it must be the directory containing
    #'tensorFieldUtilsCPP.h', 'tensorFieldUtilsCPP.cpp', 'tensorFieldUtilsPYX.pyx', etc.
    os.system('python setup.py build_ext --inplace')
    #testRigidTransformationMultiscale(15, np.array([5,5]),6)
    #testRigidTransformationMultiScale3D(np.array([6.0,0,0, 0,0,0]),6)
    #betaGT=np.array([5.0, -4.0, 3.0, 7.0, 5.5, 2.0])
    betaGT=np.array([14.0, -14.0, 14.0, 7.0, 5.5, 2.0])
    #betaGT=np.array([25.0, -24.0, 23.0, 27.0, 25.5, 22.0])
    #betaGT=np.array([13.0, -12.0, 11.0, 0.0, 0.5, 1.0])
    #betaGT=np.array([15.0, -15.0, 15.0, 1.0, -1.5, 1.0])
    #betaGT=np.array([3.0, -2.0, 1.0, 0.0, 0.5, 1.0])
    #betaGT=np.array([.0, .0, .0, 0.0, 0.0, 0.0])
    #betaGT=np.array([8.0, -6.0, 5.5, 0.0, 0.5, 1.0])
    level=4
    #testRigidTransformationMultiScale3D(betaGT,level)
    rrigid.testMultimodalRigidTransformationMultiScale3D(betaGT, level)
    #testMultimodalRigidTransformationMultiScale3D_opposite(betaGT, level)
    #testNipyRegistration(betaGT)
    #fname0='data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz'
    #fname1='data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz'
    #ofname='output.nii.gz'
    #rrigid.testIntersubjectRigidRegistration(fname0, fname1, level, ofname)


#sudo ./ANTS 3 -m MSQ[fixed.nii, moving.nii,1,0] -o test --rigid-affine true
#---affine registration using flirt:
#flirt -ref data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz -in data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz -omat M.txt -out ibsr01to02.nii

#---applying an affine registration using flirt---
#flirt -applyxfm -init output.mat -in moving.nii -ref fixed.nii -out warped.nii




#def quickTest(lambdaParam):
#    fname0='/home/khayyam/Desktop/Monchy/Data/GOOD/b0_brain.nii.gz'
#    fname1='/home/khayyam/Desktop/Monchy/Data/GOOD/t1_brain_nonlin_transformWarp.nii.gz'
#    #fname0='/home/khayyam/Desktop/Monchy/Data/GOOD/b0_brain.nii.gz'
#    #fname1='/home/khayyam/Desktop/Monchy/Data/GOOD/t1_brain_on_upsamp_b0_brain.nii.gz'
#    nib_left = nib.load(fname0)
#    nib_right = nib.load(fname1)
#    left=nib_left.get_data().squeeze()
#    right=nib_right.get_data().squeeze()
#    sl=left.shape
#    sr=right.shape
#    level=5
#    #---sagital---
#    #left=left[sl[0]//2,:,:].copy()
#    #right=right[sr[0]//2,:,:].copy()
#    #---coronal---
#    left=left[:,sl[1]//2,:].copy()
#    right=right[:,sr[1]//2,:].copy()
#    #---axial---
#    #left=left[:,:,sl[2]//2].copy()
#    #right=right[:,:,sr[2]//2].copy()
#    maskLeft=left>0
#    maskRight=right>0
#    leftPyramid=[img for img in pyramid_gaussian_2D(left, level, maskLeft)]
#    rightPyramid=[img for img in pyramid_gaussian_2D(right, level, maskRight)]
#    plotOverlaidPyramids(leftPyramid, rightPyramid)
#    displacementList=[]
#    displacement=estimateMonomodalDeformationField2DMultiScale(leftPyramid, rightPyramid, lambdaParam,0,displacementList)
#    warpPyramid=[warpImage(leftPyramid[i], displacementList[i]) for i in range(level+1)]
#    plotOverlaidPyramids(warpPyramid, rightPyramid)
#    plt.figure()
#    overlayImages(warpPyramid[0], rightPyramid[0])
#    plotDeformationField(displacement)
#    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
#    maxNorm=np.max(nrm)
#    displacement[...,0]*=(maskLeft + maskRight)
#    displacement[...,1]*=(maskLeft + maskRight)
#    plotDeformationField(displacement)
#    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
#    plt.figure()
#    plt.imshow(nrm)
#    print 'Max global displacement: ', maxNorm
