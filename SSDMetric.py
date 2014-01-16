import numpy as np
import scipy as sp
import tensorFieldUtils as tf
from SimilarityMetric import SimilarityMetric
import registrationCommon as rcommon
import matplotlib.pyplot as plt
def vCycle2D(n, k, deltaField, sigmaField, gradientField, lambdaParam, displacement):
    #presmoothing
    for i in range(k):
        error=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, displacement, None)
    if n==0:
        return error
    #solve at coarcer grid
    subSigmaField=None
    if sigmaField!=None:
        subSigmaField=tf.downsample_scalar_field(sigmaField)
    subDeltaField=tf.downsample_scalar_field(deltaField)
    subGradientField=tf.downsample_displacement_field(gradientField)
    subDisplacement=tf.downsample_displacement_field(displacement)
    subLambdaParam=0.25*lambdaParam
    vCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField, subLambdaParam, subDisplacement)
    displacement=np.array(tf.upsample_displacement_field(subDisplacement, np.array(displacement.shape).astype(np.int32)))
    #post-smoothing
    for i in range(k):
        error=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, displacement, None)
    return displacement

def vCycle3D(n, k, deltaField, sigmaField, gradientField, lambdaParam, displacement):
    #presmoothing
    for i in range(k):
        error=tf.iterateDisplacementField3DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, displacement, None)
    if n==0:
        return error
    #solve at coarcer grid
    subSigmaField=None
    if sigmaField!=None:
        subSigmaField=tf.downsample_scalar_field3D(sigmaField)
    subDeltaField=tf.downsample_scalar_field3D(deltaField)
    subGradientField=tf.downsample_displacement_field3D(gradientField)
    subDisplacement=tf.downsample_displacement_field3D(displacement)
    subLambdaParam=0.25*lambdaParam
    vCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField, subLambdaParam, subDisplacement)
    displacement=tf.upsample_displacement_field3D(subDisplacement, np.array(displacement.shape).astype(np.int32))
    #post-smoothing
    for i in range(k):
        error=tf.iterateDisplacementField3DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, displacement, None)
    return displacement

def wCycle2D(n, k, deltaField, sigmaField, gradientField, lambdaParam, displacement):
    iterFactor=2**n
    #presmoothing
    for i in range(k*iterFactor):
        error=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, displacement, None)
    if n==0:
        return error
    #solve at coarcer grid
    subSigmaField=None
    if sigmaField!=None:
        subSigmaField=tf.downsample_scalar_field(sigmaField)
    subDeltaField=tf.downsample_scalar_field(deltaField)
    subGradientField=np.array(tf.downsample_displacement_field(gradientField))
    subDisplacement=np.array(tf.downsample_displacement_field(displacement))
    subLambdaParam=lambdaParam*0.25
    wCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField, subLambdaParam, subDisplacement)
    displacement=np.array(tf.upsample_displacement_field(subDisplacement, np.array(displacement.shape).astype(np.int32)))
    #post-smoothing
    for i in range(k*iterFactor):
        error=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, displacement, None)
    #second coarcer step
    subDisplacement=np.array(tf.downsample_displacement_field(displacement))
    wCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField, subLambdaParam, subDisplacement)
    displacement=np.array(tf.upsample_displacement_field(subDisplacement, np.array(displacement.shape).astype(np.int32)))
    #second post-smoothing
    for i in range(k*iterFactor):
        error=tf.iterateDisplacementField2DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, displacement, None)
    return displacement
    
def wCycle3D(n, k, deltaField, sigmaField, gradientField, lambdaParam, displacement):
    iterFactor=4**n
    #iterFactor=1
    #presmoothing
    for i in range(k*iterFactor):
        error=tf.iterateDisplacementField3DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, displacement, None)
    if n==0:
        return error
    #solve at coarcer grid
    subSigmaField=None
    if sigmaField!=None:
        subSigmaField=tf.downsample_scalar_field3D(sigmaField)
    subDeltaField=tf.downsample_scalar_field3D(deltaField)
    subGradientField=np.array(tf.downsample_displacement_field3D(gradientField))
    subDisplacement=np.array(tf.downsample_displacement_field3D(displacement))
    subLambdaParam=lambdaParam*0.25
    wCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField, subLambdaParam, subDisplacement)
    displacement=np.array(tf.upsample_displacement_field3D(subDisplacement, np.array(displacement.shape).astype(np.int32)))
    #post-smoothing
    for i in range(k*iterFactor):
        error=tf.iterateDisplacementField3DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, displacement, None)
    #second coarcer step
    subDisplacement=np.array(tf.downsample_displacement_field3D(displacement))
    wCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField, subLambdaParam, subDisplacement)
    displacement=np.array(tf.upsample_displacement_field3D(subDisplacement, np.array(displacement.shape).astype(np.int32)))
    #second post-smoothing
    for i in range(k*iterFactor):
        error=tf.iterateDisplacementField3DCYTHON(deltaField, sigmaField, gradientField,  lambdaParam, displacement, None)
    return displacement

class SSDMetric(SimilarityMetric):
    GAUSS_SEIDEL_STEP=0
    DEMONS_STEP=1
    def getDefaultParameters(self):
        return {'lambda':1.0, 'maxInnerIter':5, 'scale':1, 'maxStepLength':0.25, 
                'sigmaDiff':3.0, 'stepType':0, 'symmetric':False}

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
        #lambdaParam=self.parameters['lambda']*(0.25**self.levelsAbove)
        lambdaParam=self.parameters['lambda']
        maxStepLength=self.parameters['maxStepLength']
        sh=self.fixedImage.shape if forwardStep else self.movingImage.shape
        deltaField=self.fixedImage-self.movingImage if forwardStep else self.movingImage - self.fixedImage
        gradient=self.gradientMoving+self.gradientFixed
        displacement=np.zeros(shape=(sh)+(self.dim,), dtype=np.float64)
        if self.dim==2:
            displacement=wCycle2D(self.levelsBelow, maxInnerIter, deltaField, None, gradient, lambdaParam, displacement)
        else:
            displacement=wCycle3D(self.levelsBelow, maxInnerIter, deltaField, None, gradient, lambdaParam, displacement)
        maxNorm=np.sqrt(np.sum(displacement**2,-1)).max()
        if maxNorm>maxStepLength:
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