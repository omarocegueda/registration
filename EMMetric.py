import numpy as np
import scipy as sp
import tensorFieldUtils as tf
from SimilarityMetric import SimilarityMetric
import matplotlib.pyplot as plt
class EMMetric(SimilarityMetric):
    GAUSS_SEIDEL_STEP=0
    DEMONS_STEP=1
    def getDefaultParameters(self):
        return {'lambda':1.0, 'maxInnerIter':200, 'innerTolerance':1e-4, 
                'scale':1, 'maxStepLength':0.25, 'sigmaDiff':3.0, 'stepType':0,
                'qLevels':256, 'symmetric':False}

    def __init__(self, parameters):
        super(EMMetric, self).__init__(parameters)
        self.setSymmetric(self.parameters['symmetric'])
        self.stepType=self.parameters['stepType']
        self.quantizationLevels=self.parameters['qLevels']
        self.fixedImageMask=None
        self.movingImageMask=None

    def initializeIteration(self):
        samplingMask=self.fixedImageMask*self.movingImageMask
        plt.figure()
        plt.imshow(samplingMask)
        if self.dim==2:
            fixedQ, grayLevels, hist=tf.quantizePositiveImageCYTHON(self.fixedImage, self.quantizationLevels)
            fixedQ=np.array(fixedQ, dtype=np.int32)
            fixedQMeans, fixedQVariances=tf.computeMaskedImageClassStatsCYTHON(samplingMask, self.movingImage, self.quantizationLevels, fixedQ)        
        else:
            fixedQ, grayLevels, hist=tf.quantizePositiveVolumeCYTHON(self.fixedImage, self.quantizationLevels)
            fixedQ=np.array(fixedQ, dtype=np.int32)
            fixedQMeans, fixedQVariances=tf.computeMaskedImageClassStatsCYTHON(samplingMask, self.movingImage, self.quantizationLevels, fixedQ)        
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
        if not self.symmetric:#Quantization of the moving image and its corresponding statistics are used only for the backward step
            return
        if self.dim==2:
            movingQ, grayLevels, hist=tf.quantizePositiveImageCYTHON(self.movingImage, self.quantizationLevels)
            movingQ=np.array(movingQ, dtype=np.int32)
            movingQMeans, movingQVariances=tf.computeMaskedImageClassStatsCYTHON(samplingMask, self.fixedImage, self.quantizationLevels, movingQ)        
        else:
            movingQ, grayLevels, hist=tf.quantizePositiveVolumeCYTHON(self.movingImage, self.quantizationLevels)
            movingQ=np.array(movingQ, dtype=np.int32)
            movingQMeans, movingQVariances=tf.computeMaskedVolumeClassStatsCYTHON(samplingMask, self.fixedImage, self.quantizationLevels, movingQ)        
        movingQMeans[0]=0
        movingQMeans=np.array(movingQMeans)
        movingQVariances=np.array(movingQVariances)
        self.movingQSigmaField=movingQVariances[movingQ]
        self.movingQMeansField=movingQMeans[movingQ]

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
        tolerance=self.parameters['innerTolerance']
        lambdaParam=self.parameters['lambda']
        maxStepLength=self.parameters['maxStepLength']
        sh=self.fixedImage.shape if forwardStep else self.movingImage.shape
        deltaField=self.fixedQMeansField-self.movingImage if forwardStep else self.movingQMeansField - self.fixedImage
        sigmaField=self.fixedQSigmaField if forwardStep else self.movingQSigmaField
        gradient=self.gradientMoving+self.gradientFixed
        displacement=np.zeros(shape=(sh)+(self.dim,), dtype=np.float64)
        error=1+tolerance
        innerIter=0
        if self.dim==2:
            while((error>tolerance)and(innerIter<maxInnerIter)):
                innerIter+=1
                error=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradient,  lambdaParam, displacement, None)
            maxNorm=np.sqrt(np.sum(displacement**2,2)).max()
            #if maxNorm>maxStepLength:
            displacement*=maxStepLength/maxNorm
        else:
            while((error>tolerance)and(innerIter<maxInnerIter)):
                innerIter+=1
                error=tf.iterateDisplacementField3DCYTHON(deltaField, sigmaField, gradient,  lambdaParam, displacement, None)
            maxNorm=np.sqrt(np.sum(displacement**2,3)).max()
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
        if direction==1:
            self.movingImageMask=transformation.warpForwardNN(movingImageMask)
        else:
            self.movingImageMask=transformation.warpBackwardNN(movingImageMask)
