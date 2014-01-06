import numpy as np
import matplotlib.pyplot as plt
import registrationCommon as rcommon
import tensorFieldUtils as tf
import UpdateRule
from TransformationModel import TransformationModel
from SSDMetric import SSDMetric

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
        if affineFixed:
            self.dim=affineFixed.shape[1]-1
        self.affineFixed=affineFixed

    def setAffineMoving(self, affineMoving):
        if affineMoving:
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
        self.movingPyramid=[img for img in rcommon.pyramid_gaussian_2D(self.moving, self.levels-1, np.ones_like(self.moving))]
        maskFixed=self.fixed>0
        self.fixedPyramid=[img for img in rcommon.pyramid_gaussian_2D(self.fixed, self.levels-1, np.ones_like(self.fixed))]
        del maskFixed
        startingForward=np.zeros(shape=self.fixedPyramid[self.levels-1].shape+(self.dim,), dtype=np.float64)
        self.forwardModel.scaleAffines(0.5**(self.levels-1))
        self.forwardModel.setForward(startingForward)
        
        startingBackward=np.zeros(shape=self.movingPyramid[self.levels-1].shape+(self.dim,), dtype=np.float64)
        self.backwardModel.scaleAffines(0.5**(self.levels-1))
        self.backwardModel.setForward(startingBackward)

    def __endOptimizer(self):
        del self.movingPyramid
        del self.fixedPyramid

    def __iterate_asymmetric(self, showImages=False):
        wmoving=self.forwardModel.warpForward(self.currentMoving)
        self.similarityMetric.setMovingImage(wmoving)
        self.similarityMetric.setFixedImage(self.currentFixed)
        self.similarityMetric.initializeIteration()
        fw=self.similarityMetric.computeForward()
        forward, meanDifference=self.updateRule.update(fw, self.forwardModel.getForward())
        self.forwardModel.setForward(forward)
        if showImages:
            plt.figure()
            rcommon.overlayImages(wmoving, self.currentFixed, False)
        return meanDifference

    def __optimize_asymmetric(self):
        self.__initOptimizer()
        for level in range(self.levels-1, -1, -1):
            print 'Processing level', level
            self.currentFixed=self.fixedPyramid[level]
            self.currentMoving=self.movingPyramid[level]
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
        wmoving=self.forwardModel.warpForward(self.currentMoving)
        wfixed=self.backwardModel.warpForward(self.currentFixed)
        self.similarityMetric.setMovingImage(wmoving)
        self.similarityMetric.setFixedImage(wfixed)
        self.similarityMetric.initializeIteration()
        fw=self.similarityMetric.computeForward()
        bw=self.similarityMetric.computeBackward()
        forward, mdForward=self.updateRule.update(fw, self.forwardModel.getForward())
        backward, mdBackward=self.updateRule.update(bw, self.backwardModel.getForward())
        invForward=np.array(tf.invert_vector_field(forward, 0.5, 100, 1e-6))
        invBackward=np.array(tf.invert_vector_field(backward, 0.5, 100, 1e-6))
        forward=np.array(tf.invert_vector_field(invForward, 0.5, 100, 1e-6))
        backward=np.array(tf.invert_vector_field(invBackward, 0.5, 100, 1e-6))
        #invForward=np.array(tf.invert_vector_field_fixed_point(forward, 50, 1e-6))
        #invBackward=np.array(tf.invert_vector_field_fixed_point(backward, 50, 1e-6))
        #forward=np.array(tf.invert_vector_field_fixed_point(invForward, 50, 1e-6))
        #backward=np.array(tf.invert_vector_field_fixed_point(invBackward, 50, 1e-6))
        self.forwardModel.setForward(forward)
        self.forwardModel.setBackward(invForward)
        self.backwardModel.setForward(backward)
        self.backwardModel.setBackward(invBackward)
        if showImages:
            plt.figure()
            rcommon.overlayImages(wmoving, wfixed, False)
        return mdForward+mdBackward

    def __optimize_symmetric(self):
        self.__initOptimizer()
        for level in range(self.levels-1, -1, -1):
            print 'Processing level', level
            self.currentFixed=self.fixedPyramid[level]
            self.currentMoving=self.movingPyramid[level]
            if level<self.levels-1:
                self.forwardModel.upsample(self.currentFixed.shape, self.currentMoving.shape)
                self.backwardModel.upsample(self.currentMoving.shape, self.currentFixed.shape)
            error=1+self.tolerance
            niter=0
            while (niter<self.maxIter[level]) and (self.tolerance<error):
                niter+=1
                error=self.__iterate_symmetric()
                if(niter==self.maxIter[level] or error<=self.tolerance):
                    error=self.__iterate_symmetric(True)
        phi1=self.forwardModel.getForward()
        phi2=self.backwardModel.getBackward()
        phi1Inv=self.forwardModel.getBackward()
        phi2Inv=self.backwardModel.getForward()
        phi, md=self.updateRule.update(phi1, phi2)
        phiInv, mdInv=self.updateRule.update(phi2Inv, phi1Inv)
        self.forwardModel.setForward(phi)
        self.forwardModel.setBackward(phiInv)
        self.__endOptimizer()

    def optimize(self):
        self.__optimize_symmetric()

    def getForward(self):
        return self.forwardModel.getForward()

def testRegistrationOptimizer2D():
    fname0='data/circle.png'
    fname1='data/C.png'
    nib_moving=plt.imread(fname0)
    nib_fixed=plt.imread(fname1)
    moving=nib_moving[:,:,0].astype(np.float64)
    fixed=nib_fixed[:,:,1].astype(np.float64)
    moving=np.copy(moving, order='C')
    fixed=np.copy(fixed, order='C')
    moving=(moving-moving.min())/(moving.max() - moving.min())
    fixed=(fixed-fixed.min())/(fixed.max() - fixed.min())
    ################Configure and run the Optimizer#####################
    maxIter=[i for i in [50,100,100,100]]
    similarityMetric=SSDMetric({'lambda':5.0})
    similarityMetric.updateType=SSDMetric.DEMONS_STEP
    #updateRule=UpdateRule.Addition()
    updateRule=UpdateRule.Composition()
    #updateRule=UpdateRule.ProjectedComposition()
    registrationOptimizer=RegistrationOptimizer(fixed, moving, None, None, similarityMetric, updateRule, maxIter)
    registrationOptimizer.optimize()
    #######################show results#################################
    displacement=registrationOptimizer.getForward()
    X1,X0=np.mgrid[0:displacement.shape[0], 0:displacement.shape[1]]
    detJacobian=rcommon.computeJacobianField(displacement)
    plt.figure()
    plt.imshow(detJacobian)
    CS=plt.contour(X0,X1,detJacobian,levels=[0.0], colors='b')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('det(J(displacement))')
    print 'J range:', '[', detJacobian.min(), detJacobian.max(),']'
    directInverse=np.array(tf.invert_vector_field(displacement, 2.0, 500, 1e-7))
    detJacobianInverse=rcommon.computeJacobianField(directInverse)
    plt.figure()
    plt.imshow(detJacobianInverse)
    CS=plt.contour(X0,X1,detJacobianInverse, levels=[0.0],colors='w')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('det(J(displacement^-1))')
    print 'J^-1 range:', '[', detJacobianInverse.min(), detJacobianInverse.max(),']'
    directResidual,stats=tf.compose_vector_fields(displacement, directInverse)
    directResidual=np.array(directResidual)
    rcommon.plotDiffeomorphism(displacement, directInverse, directResidual, 'inv-direct', 7)

if __name__=='__main__':
    testRegistrationOptimizer2D()
    
    import nibabel as nib
    result=nib.load('data/circleToC.nii.gz')
    result=result.get_data().astype(np.double)
    plt.imshow(result)