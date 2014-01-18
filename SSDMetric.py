import numpy as np
import scipy as sp
import tensorFieldUtils as tf
from SimilarityMetric import SimilarityMetric
import registrationCommon as rcommon
import matplotlib.pyplot as plt

def singleCycle2D(n, k, deltaField, sigmaField, gradientField, lambdaParam, displacement, depth=0):
    iterFactor=1
    if n==0:
        for i in range(k*iterFactor):
            error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  None, lambdaParam, displacement)
            if depth==0:
                energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
                print 'Energy after top-level iter',i+1,' [unique]:',energy
        return error
    #solve at coarcer grid
    subSigmaField=None
    if sigmaField!=None:
        subSigmaField=tf.downsample_scalar_field(sigmaField)
    subDeltaField=tf.downsample_scalar_field(deltaField)
    subGradientField=np.array(tf.downsample_displacement_field(gradientField))
    sh=np.array(displacement.shape).astype(np.int32)
    subDisplacement=np.zeros(shape=((sh[0]+1)//2, (sh[1]+1)//2, 2 ), dtype=np.float64)
    subLambdaParam=lambdaParam*0.25
    singleCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField, subLambdaParam, subDisplacement, depth+1)
    displacement+=np.array(tf.upsample_displacement_field(subDisplacement, sh))
    if depth==0:
        energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing    
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  None, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    return error

def vCycle2D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam, displacement, depth=0):
    iterFactor=1
    #presmoothing
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    if n==0:
        return error
    #solve at coarcer grid
    residual=None
    residual=tf.compute_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement, residual)
    subResidual=np.array(tf.downsample_displacement_field(residual))
    del residual
    subSigmaField=None
    if sigmaField!=None:
        subSigmaField=tf.downsample_scalar_field(sigmaField)
    subDeltaField=tf.downsample_scalar_field(deltaField)
    subGradientField=np.array(tf.downsample_displacement_field(gradientField))
    sh=np.array(displacement.shape).astype(np.int32)
    #subDisplacement=np.array(tf.downsample_displacement_field(displacement))
    subDisplacement=np.zeros(shape=((sh[0]+1)//2, (sh[1]+1)//2, 2 ), dtype=np.float64)
    subLambdaParam=lambdaParam*0.25
    vCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField, subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+=np.array(tf.upsample_displacement_field(subDisplacement, sh))
    if depth==0:
        energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing    
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    return error

def wCycle2D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam, displacement, depth=0):
    iterFactor=1
    #presmoothing
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [first]:',energy
    if n==0:
        return error
    residual=tf.compute_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement, None)
    subResidual=np.array(tf.downsample_displacement_field(residual))
    del residual
    #solve at coarcer grid
    subSigmaField=None
    if sigmaField!=None:
        subSigmaField=tf.downsample_scalar_field(sigmaField)
    subDeltaField=tf.downsample_scalar_field(deltaField)
    subGradientField=np.array(tf.downsample_displacement_field(gradientField))
    sh=np.array(displacement.shape).astype(np.int32)
    #subDisplacement=np.array(tf.downsample_displacement_field(displacement))
    subDisplacement=np.zeros(shape=((sh[0]+1)//2, (sh[1]+1)//2, 2 ), dtype=np.float64)
    subLambdaParam=lambdaParam*0.25
    wCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField, subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+=np.array(tf.upsample_displacement_field(subDisplacement, sh))
    if depth==0:
        energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration[first]:',energy
    #post-smoothing (second smoothing)
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [second]:',energy
    residual=tf.compute_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement, None)
    subResidual=np.array(tf.downsample_displacement_field(residual))
    del residual
    #subDisplacement=np.array(tf.downsample_displacement_field(displacement))
    subDisplacement=np.zeros(shape=((sh[0]+1)//2, (sh[1]+1)//2, 2 ), dtype=np.float64)
    wCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField, subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+=np.array(tf.upsample_displacement_field(subDisplacement, sh))
    if depth==0:
        energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration[second]:',energy
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [third]:',energy
    return error

def singleCycle3D(n, k, deltaField, sigmaField, gradientField, lambdaParam, displacement, depth=0):
    iterFactor=1
    if n==0:
        for i in range(k*iterFactor):
            error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  None, lambdaParam, displacement)
            if depth==0:
                energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
                print 'Energy after top-level iter',i+1,' [unique]:',energy
        return error
    #solve at coarcer grid
    subSigmaField=None
    if sigmaField!=None:
        subSigmaField=tf.downsample_scalar_field3D(sigmaField)
    subDeltaField=tf.downsample_scalar_field3D(deltaField)
    subGradientField=np.array(tf.downsample_displacement_field3D(gradientField))
    sh=np.array(displacement.shape).astype(np.int32)
    subDisplacement=np.zeros(shape=((sh[0]+1)//2, (sh[1]+1)//2, (sh[2]+1)//2, 3 ), dtype=np.float64)
    subLambdaParam=lambdaParam*0.25
    singleCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField, subLambdaParam, subDisplacement, depth+1)
    displacement+=np.array(tf.upsample_displacement_field3D(subDisplacement, sh))
    if depth==0:
        energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing    
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  None, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    return error

def vCycle3D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam, displacement, depth=0):
    iterFactor=1
    #presmoothing
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    if n==0:
        return error
    #solve at coarcer grid
    residual=None
    residual=tf.compute_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement, residual)
    subResidual=np.array(tf.downsample_displacement_field3D(residual))
    del residual
    subSigmaField=None
    if sigmaField!=None:
        subSigmaField=tf.downsample_scalar_field3D(sigmaField)
    subDeltaField=tf.downsample_scalar_field3D(deltaField)
    subGradientField=np.array(tf.downsample_displacement_field3D(gradientField))
    sh=np.array(displacement.shape).astype(np.int32)
    subDisplacement=np.zeros(shape=((sh[0]+1)//2, (sh[1]+1)//2, (sh[2]+1)//2, 3 ), dtype=np.float64)
    subLambdaParam=lambdaParam*0.25
    vCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField, subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+=np.array(tf.upsample_displacement_field3D(subDisplacement, sh))
    if depth==0:
        energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing    
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    return error

def wCycle3D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam, displacement, depth=0):
    iterFactor=1
    #presmoothing
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [first]:',energy
    if n==0:
        return error
    residual=tf.compute_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement, None)
    subResidual=np.array(tf.downsample_displacement_field3D(residual))
    del residual
    #solve at coarcer grid
    subSigmaField=None
    if sigmaField!=None:
        subSigmaField=tf.downsample_scalar_field3D(sigmaField)
    subDeltaField=tf.downsample_scalar_field3D(deltaField)
    subGradientField=np.array(tf.downsample_displacement_field3D(gradientField))
    sh=np.array(displacement.shape).astype(np.int32)
    subDisplacement=np.zeros(shape=((sh[0]+1)//2, (sh[1]+1)//2, (sh[2]+1)//2, 3 ), dtype=np.float64)
    subLambdaParam=lambdaParam*0.25
    wCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField, subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+=np.array(tf.upsample_displacement_field3D(subDisplacement, sh))
    if depth==0:
        energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration[first]:',energy
    #post-smoothing (second smoothing)
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [second]:',energy
    residual=tf.compute_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement, None)
    subResidual=np.array(tf.downsample_displacement_field3D(residual))
    del residual
    subDisplacement[...]=0
    wCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField, subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+=np.array(tf.upsample_displacement_field3D(subDisplacement, sh))
    if depth==0:
        energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration[second]:',energy
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if depth==0:
            energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [third]:',energy
    return error


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