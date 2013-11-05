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
def estimateNewMonomodalDiffeomorphicField2D(moving, fixed, lambdaParam, maxOuterIter, previousDisplacement):
    '''
    Warning: in the monomodal case, the parameter lambda must be significantly lower than in the multimodal case. Try lambdaParam=1,
    as opposed as lambdaParam=150 used in the multimodal case
    '''
    epsilon=1e-4
    sh=moving.shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]
    displacement     =np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
    residuals=np.zeros(shape=(moving.shape), dtype=np.float64)
    gradientField    =np.empty(shape=(moving.shape)+(2,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(moving.shape)+(2,), dtype=np.float64)
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
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
        maxVariation=1+epsilon
        innerIter=0
        maxResidual=0
        displacement[...]=0
        maxInnerIter=1000
        while((maxVariation>epsilon)and(innerIter<maxInnerIter)):
            innerIter+=1
            maxVariation=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, totalDisplacement, displacement, residuals)
            opt=np.max(residuals)
            if(maxResidual<opt):
                maxResidual=opt
        maxDisplacement=np.max(np.abs(displacement))
        expd, invexpd=tf.vector_field_exponential(displacement)
        totalDisplacement=tf.compose_vector_fields(expd, totalDisplacement)
    print "Iter: ",innerIter, "Max lateral displacement:", maxDisplacement, "Max variation:",maxVariation, "Max residual:", maxResidual
    if(previousDisplacement!=None):
        return totalDisplacement-previousDisplacement
    return totalDisplacement


def estimateMonomodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level, displacementList):
    n=len(movingPyramid)
    if(level==(n-1)):
        #displacement=estimateNewMonomodalDeformationField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, maxOuterIter[level], None)
        displacement=estimateNewMonomodalDiffeomorphicField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, maxOuterIter[level], None)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement
    subDisplacement=estimateMonomodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, level+1, displacementList)
    sh=movingPyramid[level].shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]*0.5
    upsampled=np.empty(shape=(movingPyramid[level].shape)+(2,), dtype=np.float64)
    upsampled[:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,0], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,1], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement=estimateNewMonomodalDiffeomorphicField2D(movingPyramid[level], fixedPyramid[level], lambdaParam, maxOuterIter[level], upsampled)
    newDisplacement+=upsampled
#    expd, invexpd=tf.vector_field_exponential(newDisplacement)
#    newDisplacement=tf.compose_vector_fields(expd,upsampled)
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return np.array(newDisplacement)

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
    displacement=estimateMonomodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxIter, 0,displacementList)
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

def testCircleToCMonomodalDiffeomorphic(lambdaParam):
    fname0='/home/omar/Desktop/circle.png'
    #fname0='/home/omar/Desktop/C_trans.png'
    fname1='/home/omar/Desktop/C.png'
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
    maxOuterIter=[10,50,100,100,100,100,100,100,100]
    displacement=estimateMonomodalDiffeomorphicField2DMultiScale(movingPyramid, fixedPyramid, lambdaParam, maxOuterIter, 0,displacementList)
    warpPyramid=[rcommon.warpImage(movingPyramid[i], displacementList[i]) for i in range(level+1)]
    rcommon.plotOverlaidPyramids(warpPyramid, fixedPyramid)
    rcommon.overlayImages(warpPyramid[0], fixedPyramid[0])
    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
    maxNorm=np.max(nrm)
    displacement[...,0]*=(maskMoving + maskFixed)
    displacement[...,1]*=(maskMoving + maskFixed)
    rcommon.plotDeformationField(displacement)
    plt.title('Deformation field')
    invd=tf.invert_vector_field(displacement, 1, 1000, 1e-7)
    residualDirect=tf.compose_vector_fields(displacement, invd)
    rcommon.plotDeformationField(residualDirect)
    plt.title('Residual after direct inverse')
    expd, invexpd=tf.vector_field_exponential(displacement)
    residualExp=tf.compose_vector_fields(expd, invexpd)
    rcommon.plotDeformationField(residualExp)
    plt.title('Residual after exponential inverse')
    lattice=rcommon.drawLattice2D(16, 16, 15)
    lattice=lattice[0:256,0:256]
    warpedLattice=rcommon.warpImage(lattice, displacement)
    plt.figure()
    plt.imshow(warpedLattice, cmap=plt.cm.gray)
    plt.title('Warped lattice')
    invertedWarpedLattice=rcommon.warpImage(lattice, residualExp)
    plt.figure()
    plt.imshow(invertedWarpedLattice, cmap=plt.cm.gray)
    plt.title('Restored warped lattice (with binary exponentiation)')
    inverseWarp=rcommon.warpImage(lattice, invexpd)
    plt.figure()
    plt.imshow(inverseWarp, cmap=plt.cm.gray)
    plt.title('Inverse-warped lattice (with binary exponentiation)')
    #nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
    #plt.figure()
    #plt.imshow(nrm)
    print 'Max global displacement: ', maxNorm

if __name__=='__main__':
    testCircleToCMonomodalDiffeomorphic(1)