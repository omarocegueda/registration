import sys
import numpy as np
import scipy as sp
import tensorFieldUtils as tf
from SimilarityMetric import SimilarityMetric
import matplotlib.pyplot as plt
import registrationCommon as rcommon
from SSDMetric import singleCycle2D, vCycle2D, wCycle2D
from SSDMetric import singleCycle3D, vCycle3D, wCycle3D
class EMMetric(SimilarityMetric):
    GAUSS_SEIDEL_STEP=0
    DEMONS_STEP=1
    def getDefaultParameters(self):
        return {'lambda':1.0, 'maxInnerIter':5, 'scale':1, 
                'maxStepLength':0.25, 'sigmaDiff':3.0, 'stepType':0, 
                'qLevels':256, 'symmetric':False,'useDoubleGradient':False}

    def __init__(self, parameters):
        super(EMMetric, self).__init__(parameters)
        self.setSymmetric(self.parameters['symmetric'])
        self.stepType=self.parameters['stepType']
        self.quantizationLevels=self.parameters['qLevels']
        self.useDoubleGradient=self.parameters['useDoubleGradient']
        self.fixedImageMask=None
        self.movingImageMask=None
        self.fixedQMeansField=None
        self.movingQMeansField=None
        self.movingQLevels=None
        self.fixedQLevels=None
        

    def initializeIteration(self):
        samplingMask=self.fixedImageMask*self.movingImageMask
        self.samplingMask=samplingMask
        if self.dim==2:
            fixedQ, self.fixedQLevels, hist=tf.quantizePositiveImageCYTHON(self.fixedImage, self.quantizationLevels)
            fixedQ=np.array(fixedQ, dtype=np.int32)
            self.fixedQLevels=np.array(self.fixedQLevels)
            fixedQMeans, fixedQVariances=tf.computeMaskedImageClassStatsCYTHON(samplingMask, self.movingImage, self.quantizationLevels, fixedQ)        
        else:
            fixedQ, self.fixedQLevels, hist=tf.quantizePositiveVolumeCYTHON(self.fixedImage, self.quantizationLevels)
            fixedQ=np.array(fixedQ, dtype=np.int32)
            self.fixedQLevels=np.array(self.fixedQLevels)
            fixedQMeans, fixedQVariances=tf.computeMaskedVolumeClassStatsCYTHON(samplingMask, self.movingImage, self.quantizationLevels, fixedQ)        
        fixedQMeans[0]=0
        fixedQMeans=np.array(fixedQMeans)
        fixedQVariances=np.array(fixedQVariances)
        self.fixedQSigmaField=fixedQVariances[fixedQ]
        self.fixedQMeansField=fixedQMeans[fixedQ]
        self.gradientMoving=np.empty(shape=(self.movingImage.shape)+(self.dim,), dtype=np.float64)
        i=0
        for grad in sp.gradient(self.movingImage):
            self.gradientMoving[...,i]=grad
            i+=1
        self.gradientFixed=np.empty(shape=(self.fixedImage.shape)+(self.dim,), dtype=np.float64)
        i=0
        for grad in sp.gradient(self.fixedImage):
            self.gradientFixed[...,i]=grad
            i+=1
        if self.dim==2:
            movingQ, self.movingQLevels, hist=tf.quantizePositiveImageCYTHON(self.movingImage, self.quantizationLevels)
            movingQ=np.array(movingQ, dtype=np.int32)
            self.movingQLevels=np.array(self.movingQLevels)
            movingQMeans, movingQVariances=tf.computeMaskedImageClassStatsCYTHON(samplingMask, self.fixedImage, self.quantizationLevels, movingQ)        
        else:
            movingQ, self.movingQLevels, hist=tf.quantizePositiveVolumeCYTHON(self.movingImage, self.quantizationLevels)
            movingQ=np.array(movingQ, dtype=np.int32)
            self.movingQLevels=np.array(self.movingQLevels)
            movingQMeans, movingQVariances=tf.computeMaskedVolumeClassStatsCYTHON(samplingMask, self.fixedImage, self.quantizationLevels, movingQ)        
        movingQMeans[0]=0
        movingQMeans=np.array(movingQMeans)
        movingQVariances=np.array(movingQVariances)
        self.movingQSigmaField=movingQVariances[movingQ]
        self.movingQMeansField=movingQMeans[movingQ]
        if self.useDoubleGradient:
            i=0
            for grad in sp.gradient(self.fixedQMeansField):
                self.gradientMoving[...,i]+=grad
                i+=1
            i=0
            for grad in sp.gradient(self.movingQMeansField):
                self.gradientFixed[...,i]=grad
                i+=1

    def computeForward(self):
        if self.stepType==EMMetric.GAUSS_SEIDEL_STEP:
            return self.computeGaussSeidelStep(True)
        elif self.stepType==EMMetric.DEMONS_STEP:
            return self.computeDemonsStep(True)
        return None

    def computeBackward(self):
        if self.stepType==EMMetric.GAUSS_SEIDEL_STEP:
            return self.computeGaussSeidelStep(False)
        elif self.stepType==EMMetric.DEMONS_STEP:
            return self.computeDemonsStep(False)
        return None

    def computeGaussSeidelStep(self, forwardStep=True):
        maxInnerIter=self.parameters['maxInnerIter']
        #lambdaParam=self.parameters['lambda']*(0.5**self.levelsAbove)
        lambdaParam=self.parameters['lambda']
        maxStepLength=self.parameters['maxStepLength']
        sh=self.fixedImage.shape if forwardStep else self.movingImage.shape
        deltaField=self.fixedQMeansField-self.movingImage if forwardStep else self.movingQMeansField - self.fixedImage
        sigmaField=self.fixedQSigmaField if forwardStep else self.movingQSigmaField
        gradient=self.gradientMoving if forwardStep else self.gradientFixed
        displacement=np.zeros(shape=(sh)+(self.dim,), dtype=np.float64)
        if self.dim==2:
            #singleCycle2D(self.levelsBelow, maxInnerIter, deltaField, sigmaField, gradient, lambdaParam, displacement)
            #self.energy=vCycle2D(self.levelsBelow, maxInnerIter, deltaField, sigmaField, gradient, None, lambdaParam, displacement)
            self.energy=wCycle2D(self.levelsBelow, maxInnerIter, deltaField, sigmaField, gradient, None, lambdaParam, displacement)
        else:
            #singleCycle2D(self.levelsBelow, maxInnerIter, deltaField, sigmaField, gradient, lambdaParam, displacement)
            self.energy=vCycle3D(self.levelsBelow, maxInnerIter, deltaField, sigmaField, gradient, None, lambdaParam, displacement)
            #wCycle3D(self.levelsBelow, maxInnerIter, deltaField, sigmaField, gradient, None, lambdaParam, displacement)
        maxNorm=np.sqrt(np.sum(displacement**2,-1)).max()
        if maxNorm>maxStepLength:
            displacement*=maxStepLength/maxNorm
        return displacement

    def computeDemonsStep(self, forwardStep=True):
        return NotImplemented

    def setStepType(self, stepType):
        self.stepType=stepType

    def getEnergy(self):
        return NotImplemented

    def setSymmetric(self, symmetric=True):
        self.symmetric=symmetric

    def useOriginalFixedImage(self, originalFixedImage):
        '''
        EMMetric computes the object mask by thresholding the original fixed
        image
        '''
        pass

    def useOriginalMovingImage(self, originalMovingImage):
        '''
        EMMetric computes the object mask by thresholding the original moving
        image
        '''
        pass

    def useFixedImageDynamics(self, originalFixedImage, transformation, direction):
        '''
        EMMetric takes advantage of the image dynamics by computing the
        current fixed image mask from the originalFixedImage mask (warped
        by nearest neighbor interpolation)
        '''
        fixedImageMask=(originalFixedImage>0).astype(np.int32)
        if transformation==None:
            self.fixedImageMask=fixedImageMask
            return
        if direction==1:
            self.fixedImageMask=transformation.warpForwardNN(fixedImageMask)
        else:
            self.fixedImageMask=transformation.warpBackwardNN(fixedImageMask)

    def useMovingImageDynamics(self, originalMovingImage, transformation, direction):
        '''
        EMMetric takes advantage of the image dynamics by computing the
        current moving image mask from the originalMovingImage mask (warped
        by nearest neighbor interpolation)
        '''
        movingImageMask=(originalMovingImage>0).astype(np.int32)
        if transformation==None:
            self.movingImageMask=movingImageMask
            return
        if direction==1:
            self.movingImageMask=transformation.warpForwardNN(movingImageMask)
        else:
            self.movingImageMask=transformation.warpBackwardNN(movingImageMask)

    def reportStatus(self):
#        plt.figure()
#        plt.subplot(1,3,1)
#        plt.plot(self.fixedQLevels)
#        plt.subplot(1,3,2)
#        plt.imshow(self.samplingMask, cmap=plt.cm.gray)
#        plt.subplot(1,3,3)
#        plt.plot(self.movingQLevels)
        if self.dim==2:
            plt.figure()
            rcommon.overlayImages(self.movingQMeansField, self.fixedQMeansField, False)
