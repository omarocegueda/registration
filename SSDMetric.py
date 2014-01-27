import numpy as np
import scipy as sp
import tensorFieldUtils as tf
from SimilarityMetric import SimilarityMetric
import registrationCommon as rcommon
import matplotlib.pyplot as plt

class SSDMetric(SimilarityMetric):
    r'''
    Similarity metric for (monomodal) nonlinear image registration defined by 
    the sum of squared differences (SSD).
    '''
    GAUSS_SEIDEL_STEP=0
    DEMONS_STEP=1
    def getDefaultParameters(self):
        return {'lambda':1.0, 'maxInnerIter':5, 'scale':1, 'maxStepLength':0.25, 
                'sigmaDiff':3.0, 'stepType':0}

    def __init__(self, parameters):
        super(SSDMetric, self).__init__(parameters)
        self.stepType=self.parameters['stepType']
        self.levelsBelow=0

    def initializeIteration(self):
        r'''
        Precomputes the gradient of the input images to be used in the computation
        of the forward and backward steps.
        '''
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
        r'''
        Computes the update displacement field to be used for registration of
        the moving image towards the fixed image
        '''
        if self.stepType==SSDMetric.GAUSS_SEIDEL_STEP:
            return self.computeGaussSeidelStep(True)
        elif self.stepType==SSDMetric.DEMONS_STEP:
            return self.computeDemonsStep(True)
        return None

    def computeBackward(self):
        r'''
        Computes the update displacement field to be used for registration of
        the fixed image towards the moving image
        '''
        if self.stepType==SSDMetric.GAUSS_SEIDEL_STEP:
            return self.computeGaussSeidelStep(False)
        elif self.stepType==SSDMetric.DEMONS_STEP:
            return self.computeDemonsStep(False)
        return None

    def computeGaussSeidelStep(self, forwardStep=True):
        r'''
        Minimizes the linearized energy function defined by the sum of squared
        differences of corresponding pixels of the input images with respect
        to the displacement field.
        '''
        maxInnerIter=self.parameters['maxInnerIter']
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
        r'''
        Computes the demons step proposed by Vercauteren et al.[1] for the SSD
        metric.
        [1] Tom Vercauteren, Xavier Pennec, Aymeric Perchant, Nicholas Ayache,
            "Diffeomorphic Demons: Efficient Non-parametric Image Registration",
            Neuroimage 2009
        '''
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

    def getEnergy(self):
        return NotImplemented

    def useOriginalFixedImage(self, originalFixedImage):
        r'''
        SSDMetric does not take advantage of the original fixed image, just pass
        '''
        pass

    def useOriginalMovingImage(self, originalMovingImage):
        r'''
        SSDMetric does not take advantage of the original moving image just pass
        '''
        pass

    def useFixedImageDynamics(self, originalFixedImage, transformation, direction):
        r'''
        SSDMetric does not take advantage of the image dynamics, just pass
        '''
        pass

    def useMovingImageDynamics(self, originalMovingImage, transformation, direction):
        r'''
        SSDMetric does not take advantage of the image dynamics, just pass
        '''
        pass

    def reportStatus(self):
        plt.figure()
        rcommon.overlayImages(self.movingImage, self.fixedImage, False)

    def getMetricName(self):
        return "SSDMetric"
#######################Multigrid algorithms for SSD-like metrics#############

printEnergy=False
def singleCycle2D(n, k, deltaField, sigmaField, gradientField, lambdaParam, displacement, depth=0):
    r'''
    One-pass multi-resolution Gauss-Seidel solver: solves the SSD-like linear 
    system starting at the coarcer resolution and then refining at the finer
    resolution.
    '''
    iterFactor=1
    if n==0:
        for i in range(k*iterFactor):
            error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  None, lambdaParam, displacement)
            if printEnergy and depth==0:
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
    if printEnergy and depth==0:
        energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing    
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  None, lambdaParam, displacement)
        if printEnergy and depth==0:
            energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    try:
        energy
    except NameError:
        energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
    return energy

def vCycle2D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam, displacement, depth=0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by first
    filtering (GS-iterate) the current level, then solves for the residual
    at a coarcer resolution andfinally refines the solution at the current resolution.
    This scheme corresponds to the V-cycle proposed by Bruhn and Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation: 
            combining highest accuracy with real-time performance", 
            10th IEEE International Conference on Computer Vision, 2005. ICCV 2005.         
    '''
    iterFactor=1
    #presmoothing
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if printEnergy and depth==0:
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
    if printEnergy and depth==0:
        energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing    
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if printEnergy and depth==0:
            energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    try:
        energy
    except NameError:
        energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
    return energy

def wCycle2D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam, displacement, depth=0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by performing
    two v-cycles at each resolution, which corresponds to the w-cycle scheme
    proposed by Bruhn and Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation: 
            combining highest accuracy with real-time performance", 
            10th IEEE International Conference on Computer Vision, 2005. ICCV 2005.         
    '''
    iterFactor=1
    #presmoothing
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if printEnergy and depth==0:
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
    if printEnergy and depth==0:
        energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration[first]:',energy
    #post-smoothing (second smoothing)
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if printEnergy and depth==0:
            energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [second]:',energy
    residual=tf.compute_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement, None)
    subResidual=np.array(tf.downsample_displacement_field(residual))
    del residual
    #subDisplacement=np.array(tf.downsample_displacement_field(displacement))
    subDisplacement=np.zeros(shape=((sh[0]+1)//2, (sh[1]+1)//2, 2 ), dtype=np.float64)
    wCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField, subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+=np.array(tf.upsample_displacement_field(subDisplacement, sh))
    if printEnergy and depth==0:
        energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration[second]:',energy
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD2D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if printEnergy and depth==0:
            energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [third]:',energy
    try:
        energy
    except NameError:
        energy=tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
    return energy

def singleCycle3D(n, k, deltaField, sigmaField, gradientField, lambdaParam, displacement, depth=0):
    r'''
    One-pass multi-resolution Gauss-Seidel solver: solves the SSD-like linear 
    system starting at the coarcer resolution and then refining at the finer
    resolution.
    '''
    iterFactor=1
    if n==0:
        for i in range(k*iterFactor):
            error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  None, lambdaParam, displacement)
            if printEnergy and depth==0:
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
    if printEnergy and depth==0:
        energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing    
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  None, lambdaParam, displacement)
        if printEnergy and depth==0:
            energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    try:
        energy
    except NameError:
        energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
    return energy

def vCycle3D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam, displacement, depth=0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by first
    filtering (GS-iterate) the current level, then solves for the residual
    at a coarcer resolution andfinally refines the solution at the current resolution.
    This scheme corresponds to the V-cycle proposed by Bruhn and Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation: 
            combining highest accuracy with real-time performance", 
            10th IEEE International Conference on Computer Vision, 2005. ICCV 2005.         
    '''
    iterFactor=1
    #presmoothing
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if printEnergy and depth==0:
            energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    if n==0:
        return error
    #solve at coarcer grid
    residual=tf.compute_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement, None)
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
    del subDeltaField
    del subSigmaField
    del subGradientField
    del subResidual
    tf.accumulate_upsample_displacement_field3D(subDisplacement, displacement)
    del subDisplacement
    if printEnergy and depth==0:
        energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing    
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if printEnergy and depth==0:
            energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    try:
        energy
    except NameError:
        energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
    return energy

def wCycle3D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam, displacement, depth=0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by performing
    two v-cycles at each resolution, which corresponds to the w-cycle scheme
    proposed by Bruhn and Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation: 
            combining highest accuracy with real-time performance", 
            10th IEEE International Conference on Computer Vision, 2005. ICCV 2005.         
    '''
    iterFactor=1
    #presmoothing
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if printEnergy and depth==0:
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
    if printEnergy and depth==0:
        energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration[first]:',energy
    #post-smoothing (second smoothing)
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if printEnergy and depth==0:
            energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [second]:',energy
    residual=tf.compute_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement, None)
    subResidual=np.array(tf.downsample_displacement_field3D(residual))
    del residual
    subDisplacement[...]=0
    wCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField, subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+=np.array(tf.upsample_displacement_field3D(subDisplacement, sh))
    if printEnergy and depth==0:
        energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
        print 'Energy after low-res iteration[second]:',energy
    for i in range(k*iterFactor):
        error=tf.iterate_residual_displacement_field_SSD3D(deltaField, sigmaField, gradientField,  target, lambdaParam, displacement)
        if printEnergy and depth==0:
            energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
            print 'Energy after top-level iter',i+1,' [third]:',energy
    try:
        energy
    except NameError:
        energy=tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,  lambdaParam, displacement)
    return energy
