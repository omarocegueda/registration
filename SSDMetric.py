import numpy as np
import scipy as sp
import tensorFieldUtils as tf
from SimilarityMetric import SimilarityMetric
import registrationCommon as rcommon
import matplotlib.pyplot as plt
class SSDMetric(SimilarityMetric):
    GAUSS_SEIDEL_STEP=0
    DEMONS_STEP=1
    def getDefaultParameters(self):
        return {'lambda':1.0, 'maxInnerIter':200, 'innerTolerance':1e-4, 
                'scale':1, 'maxStepLength':0.25, 'sigmaDiff':3.0, 'stepType':0,
                'symmetric':False}

    def __init__(self, parameters):
        super(SSDMetric, self).__init__(parameters)
        self.stepType=self.parameters['stepType']
        self.setSymmetric(self.parameters['symmetric'])
        self.levelsBelow=0

    def initializeIteration(self):
        self.gradientMoving=np.empty(shape=(self.movingImage.shape)+(self.dim,), dtype=np.float64)
        i=0
        for grad in sp.gradient(self.movingImage):
            self.gradientMoving[...,i]=grad
            i+=1
        i=0
        self.gradientFixed=np.empty(shape=(self.fixedImage.shape)+(self.dim,), dtype=np.float64)
        for grad in sp.gradient(self.fixedImage):
            self.gradientFixed[...,i]=grad
            i+=1

    def computeForward(self):
        if self.stepType==SSDMetric.GAUSS_SEIDEL_STEP:
            return self.computeGaussSeidelStep(True)
        elif self.stepType==SSDMetric.DEMONS_STEP:
            return self.computeDemonsStep(True)
        return None

    def computeBackward(self):
        if not self.symmetric:
            print 'Error: SSDMetric was not set as symmetric'
            return None
        if self.stepType==SSDMetric.GAUSS_SEIDEL_STEP:
            return self.computeGaussSeidelStep(False)
        elif self.stepType==SSDMetric.DEMONS_STEP:
            return self.computeDemonsStep(False)
        return None

    def computeGaussSeidelStep(self, forwardStep=True):
        maxInnerIter=self.parameters['maxInnerIter']
        tolerance=self.parameters['innerTolerance']
        lambdaParam=self.parameters['lambda']
        maxStepLength=self.parameters['maxStepLength']
        sh=self.fixedImage.shape if forwardStep else self.movingImage.shape
        deltaField=self.fixedImage-self.movingImage if forwardStep else self.movingImage - self.fixedImage
        gradient=self.gradientMoving+self.gradientFixed
        displacement=np.zeros(shape=(sh)+(self.dim,), dtype=np.float64)
        error=1+tolerance
        innerIter=0
        if self.dim==2:
            #displacement=rcommon.vCycle2D(3, 5, deltaField, gradient, lambdaParam, displacement)
            displacement=rcommon.wCycle2D(self.levelsBelow, 5, deltaField, gradient, lambdaParam, displacement)
#            while((error>tolerance)and(innerIter<maxInnerIter)):
#                innerIter+=1
#                error=tf.iterateDisplacementField2DCYTHON(deltaField, None, gradient,  lambdaParam, displacement, None)
#            maxNorm=np.sqrt(np.sum(displacement**2,2)).max()
#            displacement*=maxStepLength/maxNorm
        else:
            while((error>tolerance)and(innerIter<maxInnerIter)):
                innerIter+=1
                error=tf.iterateDisplacementField3DCYTHON(deltaField, None, gradient,  lambdaParam, displacement, None)
            maxNorm=np.sqrt(np.sum(displacement**2,3)).max()
            displacement*=maxStepLength/maxNorm
        return displacement

    def computeDemonsStep(self, forwardStep=True):
        sigmaDiff=self.parameters['sigmaDiff']
        maxStepLength=self.parameters['maxStepLength']
        scale=self.parameters['scale']
        deltaField=self.fixedImage-self.movingImage if forwardStep else self.movingImage - self.fixedImage
        gradient=self.gradientMoving+self.gradientFixed
        if self.dim==2:
            forward=tf.compute_demons_step2D(deltaField, gradient, maxStepLength, scale)
            forward[...,0]=sp.ndimage.filters.gaussian_filter(forward[...,0], sigmaDiff)
            forward[...,1]=sp.ndimage.filters.gaussian_filter(forward[...,1], sigmaDiff)
        else:
            forward=tf.compute_demons_step2D(deltaField, gradient, maxStepLength, scale)
            forward[...,0]=sp.ndimage.filters.gaussian_filter(forward[...,0], sigmaDiff)
            forward[...,1]=sp.ndimage.filters.gaussian_filter(forward[...,1], sigmaDiff)
            forward[...,2]=sp.ndimage.filters.gaussian_filter(forward[...,2], sigmaDiff)
        return forward

    def setStepType(self, stepType):
        self.stepType=stepType

    def getEnergy(self):
        return NotImplemented

    def setSymmetric(self, symmetric=True):
        self.symmetric=symmetric

    def useOriginalFixedImage(self, originalFixedImage):
        '''
        SSDMetric does not take advantage of the original fixed image, just pass
        '''
        pass

    def useOriginalMovingImage(self, originalMovingImage):
        '''
        SSDMetric does not take advantage of the original moving image just pass
        '''
        pass

    def useFixedImageDynamics(self, originalFixedImage, transformation, direction):
        '''
        SSDMetric does not take advantage of the image dynamics, just pass
        '''
        pass

    def useMovingImageDynamics(self, originalMovingImage, transformation, direction):
        '''
        SSDMetric does not take advantage of the image dynamics, just pass
        '''
        pass

    def reportStatus(self):
        plt.figure()
        rcommon.overlayImages(self.movingImage, self.fixedImage, False)