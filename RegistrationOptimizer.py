import time
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import registrationCommon as rcommon
import tensorFieldUtils as tf
import UpdateRule
from TransformationModel import TransformationModel
from SSDMetric import SSDMetric
from EMMetric import EMMetric
#from PIL import Image, ImageSequence
#from images2gif import writeGif
from scipy import interp

class RegistrationOptimizer(object):
    def __init__(self, fixed=None, moving=None, affineFixed=None, affineMoving=None, similarityMetric=None, updateRule=None, maxIter=None):
        self.dim=0
        self.setFixedImage(fixed)
        self.setMovingImage(moving)
        self.setAffineFixed(affineFixed)
        self.setAffineMoving(affineMoving)
        self.similarityMetric=similarityMetric
        self.updateRule=updateRule
        self.setMaxIter(maxIter)
        self.tolerance=1e-6
        self.symmetric=False
        self.fixed=fixed
        self.moving=moving
        self.forwardModel=TransformationModel(None, None, affineFixed, affineMoving)
        self.backwardModel=TransformationModel(None, None, affineMoving,  affineFixed)
        self.energyList=None
        self.reportStatus=False

    def __checkReady(self):
        ready=True
        if self.fixed==None:
            ready=False
            print 'Error: Fixed image not set.'
        elif self.dim!=len(self.fixed.shape):
            ready=False
            print 'Error: inconsistent dimensions. Last dimension update: %d. Fixed image dimension: %d.'%(self.dim, len(self.fixed.shape))
        if self.moving==None:
            ready=False
            print 'Error: Moving image not set.'
        elif self.dim!=len(self.moving.shape):
            ready=False
            print 'Error: inconsistent dimensions. Last dimension update: %d. Moving image dimension: %d.'%(self.dim, len(self.moving.shape))
        if self.similarityMetric==None:
            ready=False
            print 'Error: Similarity metric not set.'
        if self.updateRule==None:
            ready=False
            print 'Error: Update rule not set.'
        if self.maxIter==None:
            ready=False
            print 'Error: Maximum number of iterations per level not set.'
        if self.affineMoving==None:
            print 'Warning: affine transformation not set for moving image. I will use the identity.'
        elif self.dim != self.affineMoving.shape[1]-1:
            ready=False
            print 'Error: inconsistent dimensions. Last dimension update: %d. Moving Affine domain: %d.'%(self.dim, self.affineMoving[1]-1)
        if self.affineFixed==None:
            print 'Warning: affine transformation not set for fixed image. I will use the identity.'
        elif self.dim != self.affineFixed.shape[1]-1:
            ready=False
            print 'Error: inconsistent dimensions. Last dimension update: %d. Fixed Affine domain: %d.'%(self.dim, self.affineFixed[1]-1)
        return ready

    def setSimilarityMetric(self, similarityMetric):
        self.similarityMetric=similarityMetric

    def setAffineFixed(self, affineFixed):
        if affineFixed!=None:
            self.dim=affineFixed.shape[1]-1
        self.affineFixed=affineFixed

    def setAffineMoving(self, affineMoving):
        if affineMoving!=None:
            self.dim=affineMoving.shape[1]-1
        self.affineMoving=affineMoving

    def setUpdateRule(self, updateRule):
        self.updateRule=updateRule

    def setFixedImage(self, fixed):
        if fixed!=None:
            self.dim=len(fixed.shape)
        self.fixed=fixed

    def setMovingImage(self, moving):
        if moving!=None:
            self.dim=len(moving.shape)
        self.moving=moving

    def setMaxIter(self, maxIter):
        self.levels=len(maxIter) if maxIter else 0    
        self.maxIter=maxIter

    def __initOptimizer(self):
        ready=self.__checkReady()
        if not ready:
            return False
        if self.dim==2:
            self.movingPyramid=[img for img in rcommon.pyramid_gaussian_2D(self.moving, self.levels-1, np.ones_like(self.moving))]
            self.fixedPyramid=[img for img in rcommon.pyramid_gaussian_2D(self.fixed, self.levels-1, np.ones_like(self.fixed))]
        else:
            self.movingPyramid=[img for img in rcommon.pyramid_gaussian_3D(self.moving, self.levels-1, np.ones_like(self.moving))]
            self.fixedPyramid=[img for img in rcommon.pyramid_gaussian_3D(self.fixed, self.levels-1, np.ones_like(self.fixed))]
        startingForward=np.zeros(shape=self.fixedPyramid[self.levels-1].shape+(self.dim,), dtype=np.float64)
        startingForwardInv=np.zeros(shape=self.fixedPyramid[self.levels-1].shape+(self.dim,), dtype=np.float64)
        self.forwardModel.scaleAffines(0.5**(self.levels-1))
        self.forwardModel.setForward(startingForward)
        self.forwardModel.setBackward(startingForwardInv)
        startingBackward=np.zeros(shape=self.movingPyramid[self.levels-1].shape+(self.dim,), dtype=np.float64)
        startingBackwardInverse=np.zeros(shape=self.fixedPyramid[self.levels-1].shape+(self.dim,), dtype=np.float64)
        self.backwardModel.scaleAffines(0.5**(self.levels-1))
        self.backwardModel.setForward(startingBackward)
        self.backwardModel.setBackward(startingBackwardInverse)

    def __endOptimizer(self):
        del self.movingPyramid
        del self.fixedPyramid

    def __iterate_asymmetric(self, showImages=False):
        wmoving=self.forwardModel.warpForward(self.currentMoving)
        self.similarityMetric.setMovingImage(wmoving)
        self.similarityMetric.useMovingImageDynamics(self.currentMoving, self.forwardModel, 1)
        self.similarityMetric.setFixedImage(self.currentFixed)
        self.similarityMetric.useFixedImageDynamics(self.currentFixed, None, 1)
        self.similarityMetric.initializeIteration()
        fw=self.similarityMetric.computeForward()
        forward, meanDifference=self.updateRule.update(fw, self.forwardModel.getForward())
        self.forwardModel.setForward(forward)
        if showImages:
            self.similarityMetric.reportStatus()
        return meanDifference

    def __optimize_asymmetric(self):
        self.__initOptimizer()
        for level in range(self.levels-1, -1, -1):
            print 'Processing level', level
            self.currentFixed=self.fixedPyramid[level]
            self.currentMoving=self.movingPyramid[level]
            self.similarityMetric.useOriginalFixedImage(self.fixedPyramid[level])
            self.similarityMetric.useOriginalMovingImage(self.movingPyramid[level])
            if level<self.levels-1:
                self.forwardModel.upsample(self.currentFixed.shape, self.currentMoving.shape)
            error=1+self.tolerance
            niter=0
            while (niter<self.maxIter[level]) and (self.tolerance<error):
                niter+=1
                error=self.__iterate_asymmetric()
                if(niter==self.maxIter[level] or error<=self.tolerance):
                    error=self.__iterate_asymmetric(True)
        self.__endOptimizer()

    def __iterate_symmetric(self, showImages=False):
        #tic=time.time()
        wmoving=self.backwardModel.warpBackward(self.currentMoving)
        wfixed=self.forwardModel.warpBackward(self.currentFixed)
        self.similarityMetric.setMovingImage(wmoving)
        self.similarityMetric.useMovingImageDynamics(self.currentMoving, self.backwardModel, -1)
        self.similarityMetric.setFixedImage(wfixed)
        self.similarityMetric.useFixedImageDynamics(self.currentFixed, self.forwardModel, -1)
        self.similarityMetric.initializeIteration()
        fw=self.similarityMetric.computeForward()
        try:
            fwEnergy=self.similarityMetric.energy
        except NameError:
            pass
        bw=self.similarityMetric.computeBackward()
        try:
            bwEnergy=self.similarityMetric.energy
        except NameError:
            pass
        try:
            n=len(self.energyList)
            print n,':\t',fwEnergy,'\t',bwEnergy,'\t',fwEnergy+bwEnergy,'\t','-' if len(self.energyList)<3 else self.__getEnergyDerivative()
            self.energyList.append(fwEnergy+bwEnergy)
        except NameError:
            pass
#        if len(self.energyList)>=22:
#            der=self.__getEnergyDerivative()
#            if(der>=0):
#                return -1
        forward, mdForward=self.updateRule.update(self.forwardModel.getForward(), fw)
        backward, mdBackward=self.updateRule.update(self.backwardModel.getForward(), bw)
        if self.dim==2:
            invForward=np.array(tf.invert_vector_field_fixed_point(forward, 20, 1e-3, self.forwardModel.getBackward()))
            invBackward=np.array(tf.invert_vector_field_fixed_point(backward, 20, 1e-3, self.backwardModel.getBackward()))
            forward=np.array(tf.invert_vector_field_fixed_point(invForward, 20, 1e-3, self.forwardModel.getForward()))
            backward=np.array(tf.invert_vector_field_fixed_point(invBackward, 20, 1e-3, self.backwardModel.getForward()))
        else:
            invForward=np.array(tf.invert_vector_field_fixed_point3D(forward, 20, 1e-3, self.forwardModel.getBackward()))
            invBackward=np.array(tf.invert_vector_field_fixed_point3D(backward, 20, 1e-3, self.backwardModel.getBackward()))
            forward=np.array(tf.invert_vector_field_fixed_point3D(invForward, 20, 1e-3, self.forwardModel.getForward()))
            backward=np.array(tf.invert_vector_field_fixed_point3D(invBackward, 20, 1e-3, self.backwardModel.getForward()))
        self.forwardModel.setForward(forward)
        self.forwardModel.setBackward(invForward)
        self.backwardModel.setForward(backward)
        self.backwardModel.setBackward(invBackward)
        if showImages:
            self.similarityMetric.reportStatus()
        #toc=time.time()
        #print('Iter time: %f sec' % (toc - tic))
        return mdForward+mdBackward

    def __getEnergyDerivative(self):
        n=len(self.energyList)
        q = np.poly1d(np.polyfit(range(n), self.energyList,2)).deriv()
        der=q(n-1.5)
        return der

    def __report_status(self):
        wmoving=self.backwardModel.warpBackward(self.currentMoving)
        wfixed=self.forwardModel.warpBackward(self.currentFixed)
        self.similarityMetric.setMovingImage(wmoving)
        self.similarityMetric.useMovingImageDynamics(self.currentMoving, self.backwardModel, -1)
        self.similarityMetric.setFixedImage(wfixed)
        self.similarityMetric.useFixedImageDynamics(self.currentFixed, self.forwardModel, -1)
        self.similarityMetric.initializeIteration()
        self.similarityMetric.reportStatus()

    def __optimize_symmetric(self):
        self.__initOptimizer()
        for level in range(self.levels-1, -1, -1):
            print 'Processing level', level
            self.currentFixed=self.fixedPyramid[level]
            self.currentMoving=self.movingPyramid[level]
            self.similarityMetric.useOriginalFixedImage(self.fixedPyramid[level])
            self.similarityMetric.useOriginalMovingImage(self.movingPyramid[level])
            self.similarityMetric.setLevelsBelow(self.levels-level)
            self.similarityMetric.setLevelsAbove(level)
            if level<self.levels-1:
                self.forwardModel.upsample(self.currentFixed.shape, self.currentMoving.shape)
                self.backwardModel.upsample(self.currentMoving.shape, self.currentFixed.shape)
            error=1+self.tolerance
            niter=0
            self.energyList=[]
            while (niter<self.maxIter[level]) and (self.tolerance<error):
                niter+=1
                error=self.__iterate_symmetric()
            if self.reportStatus:
                self.__report_status()
        phi1=self.forwardModel.getForward()
        phi2=self.backwardModel.getBackward()
        phi1Inv=self.forwardModel.getBackward()
        phi2Inv=self.backwardModel.getForward()
        phi, md=self.updateRule.update(phi1, phi2)
        phiInv, mdInv=self.updateRule.update(phi2Inv, phi1Inv)
        self.forwardModel.setForward(phi)
        self.forwardModel.setBackward(phiInv)
        residual, stats=self.forwardModel.computeInversionError()
        print 'Residual error (Symmetric diffeomorphism):',stats[1],'. (',stats[2],')'
#        try:
#            writeGif("evolution.gif", self.frames, duration=10.0, dither=0)
#        except NameError:
#            pass
        self.__endOptimizer()

    def optimize(self):
        self.__optimize_symmetric()
        #self.__optimize_asymmetric()

    def getForward(self):
        return self.forwardModel.getForward()

    def getBackward(self):
        return self.forwardModel.getBackward()

def testRegistrationOptimizerMonomodal2D():
    fnameMoving='data/circle.png'
    fnameFixed='data/C.png'
    nib_moving=plt.imread(fnameMoving)
    nib_fixed=plt.imread(fnameFixed)
    moving=nib_moving[:,:,0].astype(np.float64)
    fixed=nib_fixed[:,:,1].astype(np.float64)
    moving=np.copy(moving, order='C')
    fixed=np.copy(fixed, order='C')
    moving=(moving-moving.min())/(moving.max() - moving.min())
    fixed=(fixed-fixed.min())/(fixed.max() - fixed.min())
    ################Configure and run the Optimizer#####################
    maxIter=[i for i in [25,50,100,100]]
    similarityMetric=SSDMetric({'symmetric':True, 'lambda':5.0, 'stepType':SSDMetric.GAUSS_SEIDEL_STEP})
    updateRule=UpdateRule.Composition()
    #updateRule=UpdateRule.ProjectedComposition()
    registrationOptimizer=RegistrationOptimizer(fixed, moving, None, None, similarityMetric, updateRule, maxIter)
    registrationOptimizer.optimize()
    #######################show results#################################
    displacement=registrationOptimizer.getForward()
    directInverse=registrationOptimizer.getBackward()
    movingToFixed=np.array(tf.warp_image(moving, displacement))
    fixedToMoving=np.array(tf.warp_image(fixed, directInverse))
    rcommon.overlayImages(movingToFixed, fixed, True)
    rcommon.overlayImages(fixedToMoving, moving, True)
#    X1,X0=np.mgrid[0:displacement.shape[0], 0:displacement.shape[1]]
#    detJacobian=rcommon.computeJacobianField(displacement)
#    plt.figure()
#    plt.imshow(detJacobian)
#    CS=plt.contour(X0,X1,detJacobian,levels=[0.0], colors='b')
#    plt.clabel(CS, inline=1, fontsize=10)
#    plt.title('det(J(displacement))')
#    print 'J range:', '[', detJacobian.min(), detJacobian.max(),']'
    #directInverse=np.array(tf.invert_vector_field(displacement, 2.0, 500, 1e-7))
#    detJacobianInverse=rcommon.computeJacobianField(directInverse)
#    plt.figure()
#    plt.imshow(detJacobianInverse)
#    CS=plt.contour(X0,X1,detJacobianInverse, levels=[0.0],colors='w')
#    plt.clabel(CS, inline=1, fontsize=10)
#    plt.title('det(J(displacement^-1))')
#    print 'J^-1 range:', '[', detJacobianInverse.min(), detJacobianInverse.max(),']'
    directResidual,stats=tf.compose_vector_fields(displacement, directInverse)
    directResidual=np.array(directResidual)
    rcommon.plotDiffeomorphism(displacement, directInverse, directResidual, 'inv-direct', 7)

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

def testRegistrationOptimizerMultimodal2D(lambdaParam, synthetic):
    displacementGTName='templateToIBSR01_GT.npy'
    fnameMoving='data/t2/IBSR_t2template_to_01.nii.gz'
    fnameFixed='data/t1/IBSR_template_to_01.nii.gz'
#    fnameMoving='data/circle.png'
#    fnameFixed='data/C.png'
    nifti=True
    if nifti:
        nib_moving = nib.load(fnameMoving)
        nib_fixed = nib.load(fnameFixed)
        moving=nib_moving.get_data().squeeze().astype(np.float64)
        fixed=nib_fixed.get_data().squeeze().astype(np.float64)
        moving=np.copy(moving, order='C')
        fixed=np.copy(fixed, order='C')
        sl=moving.shape
        sr=fixed.shape    
        moving=moving[:,sl[1]//2,:].copy()
        fixed=fixed[:,sr[1]//2,:].copy()
        moving=histeq(moving)
        fixed=histeq(fixed)
        moving=(moving-moving.min())/(moving.max()-moving.min())
        fixed=(fixed-fixed.min())/(fixed.max()-fixed.min())
    else:
        nib_moving=plt.imread(fnameMoving)
        nib_fixed=plt.imread(fnameFixed)
        nib_moving=histeq(nib_moving)
        nib_fixed=histeq(nib_fixed)
        moving=nib_moving[:,:,0].astype(np.float64)
        fixed=nib_fixed[:,:,1].astype(np.float64)
        moving=np.copy(moving, order='C')
        fixed=np.copy(fixed, order='C')
        moving=(moving-moving.min())/(moving.max() - moving.min())
        fixed=(fixed-fixed.min())/(fixed.max() - fixed.min())
    #maxIter=[i for i in [25,50,100,100]]
    maxIter=[i for i in [25,50,100]]
    similarityMetric=EMMetric({'symmetric':True, 
                               'lambda':lambdaParam, 
                               'stepType':SSDMetric.GAUSS_SEIDEL_STEP, 
                               'qLevels':256, 
                               'maxInnerIter':5,
                               'useDoubleGradient':True,
                               'maxStepLength':0.25})    
    updateRule=UpdateRule.Composition()
    if(synthetic):
        print 'Generating synthetic field...'
        #----apply synthetic deformation field to fixed image
        GT=rcommon.createDeformationField2D_type2(fixed.shape[0], fixed.shape[1], 8)
        warpedFixed=rcommon.warpImage(fixed,GT)
    else:
        templateT1=nib.load('data/t1/IBSR_template_to_01.nii.gz')
        templateT1=templateT1.get_data().squeeze().astype(np.float64)
        templateT1=np.copy(templateT1, order='C')
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
            ibsrT1=np.copy(ibsrT1, order='C')
            ibsrT1=ibsrT1[:,sh[1]//2,:]
            ibsrT1=(ibsrT1-ibsrT1.min())/(ibsrT1.max()-ibsrT1.min())
            #register the template(moving) to the ibsr sample(fixed)
            #updateRule=UpdateRule.ProjectedComposition()
            registrationOptimizer=RegistrationOptimizer(ibsrT1, templateT1, None, None, similarityMetric, updateRule, maxIter)
            registrationOptimizer.optimize()
            #----apply 'realistic' deformation field to fixed image
            GT=registrationOptimizer.getForward()
            np.save(displacementGTName, GT)
        warpedFixed=rcommon.warpImage(templateT1, GT)
    print 'Registering T2 (template) to deformed T1 (template)...'
    plt.figure()
    rcommon.overlayImages(warpedFixed, moving, False)
    registrationOptimizer=RegistrationOptimizer(warpedFixed, moving, None, None, similarityMetric, updateRule, maxIter)
    registrationOptimizer.optimize()
    #######################show results#################################
    displacement=registrationOptimizer.getForward()
    directInverse=registrationOptimizer.getBackward()
    movingToFixed=np.array(tf.warp_image(moving, displacement))
    fixedToMoving=np.array(tf.warp_image(warpedFixed, directInverse))
    rcommon.overlayImages(movingToFixed, fixedToMoving, True)
#    X1,X0=np.mgrid[0:displacement.shape[0], 0:displacement.shape[1]]
#    detJacobian=rcommon.computeJacobianField(displacement)
#    plt.figure()
#    plt.imshow(detJacobian)
#    CS=plt.contour(X0,X1,detJacobian,levels=[0.0], colors='b')
#    plt.clabel(CS, inline=1, fontsize=10)
#    plt.title('det(J(displacement))')
#    print 'J range:', '[', detJacobian.min(), detJacobian.max(),']'
    #directInverse=np.array(tf.invert_vector_field(displacement, 2.0, 500, 1e-7))
#    detJacobianInverse=rcommon.computeJacobianField(directInverse)
#    plt.figure()
#    plt.imshow(detJacobianInverse)
#    CS=plt.contour(X0,X1,detJacobianInverse, levels=[0.0],colors='w')
#    plt.clabel(CS, inline=1, fontsize=10)
#    plt.title('det(J(displacement^-1))')
#    print 'J^-1 range:', '[', detJacobianInverse.min(), detJacobianInverse.max(),']'
    directResidual,stats=tf.compose_vector_fields(displacement, directInverse)
    directResidual=np.array(directResidual)
    rcommon.plotDiffeomorphism(displacement, directInverse, directResidual, 'inv-direct', 7)
    
    residual=((displacement-GT))**2
    meanDisplacementError=np.sqrt(residual.sum(2)*(warpedFixed>0)).mean()
    stdevDisplacementError=np.sqrt(residual.sum(2)*(warpedFixed>0)).std()
    print 'Mean displacement error: ', meanDisplacementError,'(',stdevDisplacementError,')'

if __name__=='__main__':
    tic=time.time()
    testRegistrationOptimizerMultimodal2D(50, True)
    toc=time.time()
    print('Registration time: %f sec' % (toc - tic))
    #testRegistrationOptimizerMonomodal2D()
    
#    import nibabel as nib
#    result=nib.load('data/circleToC.nii.gz')
#    result=result.get_data().astype(np.double)
#    plt.imshow(result)