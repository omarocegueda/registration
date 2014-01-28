import numpy as np
import scipy as sp
import tensorFieldUtils as tf
from SimilarityMetric import SimilarityMetric
import matplotlib.pyplot as plt
import registrationCommon as rcommon
import SSDMetric
class EMMetric(SimilarityMetric):
    r'''
    Similarity metric based on the Expectation-Maximization algorithm to handle
    multi-modal images. The transfer function is modeled as a set of hidden
    random variables that are estimated at each iteration of the algorithm.
    '''
    GAUSS_SEIDEL_STEP=0
    DEMONS_STEP=1
    SINGLECYCLE_ITER=0
    VCYCLE_ITER=1
    WCYCLE_ITER=2
    def getDefaultParameters(self):
        return {'lambda':1.0, 'maxInnerIter':5, 'scale':1, 
                'maxStepLength':0.25, 'sigmaDiff':3.0, 'stepType':0, 
                'qLevels':256,'useDoubleGradient':True,
                'iterationType':'vCycle'}

    def __init__(self, parameters):
        super(EMMetric, self).__init__(parameters)
        self.stepType=self.parameters['stepType']
        self.quantizationLevels=self.parameters['qLevels']
        self.useDoubleGradient=self.parameters['useDoubleGradient']
        self.fixedImageMask=None
        self.movingImageMask=None
        self.fixedQMeansField=None
        self.movingQMeansField=None
        self.movingQLevels=None
        self.fixedQLevels=None
        if self.parameters['iterationType']=='singleCycle':
            self.iterationType=EMMetric.SINGLECYCLE_ITER
        elif self.parameters['iterationType']=='wCycle':
            self.iterationType=EMMetric.WCYCLE_ITER
        else:
            self.iterationType=EMMetric.VCYCLE_ITER

    def __connectFunctions(self):
        if self.dim==2:
            self.quantize=tf.quantizePositiveImageCYTHON
            self.computeStats=tf.computeMaskedImageClassStatsCYTHON
            if self.iterationType==EMMetric.SINGLECYCLE_ITER:
                self.multiResolutionIteration=SSDMetric.singleCycle2D
            elif self.iterationType==EMMetric.VCYCLE_ITER:
                self.multiResolutionIteration=SSDMetric.vCycle2D
            else:
                self.multiResolutionIteration=SSDMetric.wCycle2D
        else:
            self.quantize=tf.quantizePositiveVolumeCYTHON
            self.computeStats=tf.computeMaskedVolumeClassStatsCYTHON
            if self.iterationType==EMMetric.SINGLECYCLE_ITER:
                self.multiResolutionIteration=SSDMetric.singleCycle3D
            elif self.iterationType==EMMetric.VCYCLE_ITER:
                self.multiResolutionIteration=SSDMetric.vCycle3D
            else:
                self.multiResolutionIteration=SSDMetric.wCycle3D
        if self.stepType==EMMetric.DEMONS_STEP:
            self.computeStep=self.computeDemonsStep
        else:
            self.computeStep=self.computeGaussSeidelStep

    def initializeIteration(self):
        r'''
        Precomputes the transfer functions (hidden random variables) and 
        variances of the estimators. Also precomputes the gradient of both
        input images. Note that once the images are transformed to the opposite
        modality, the gradient of the transformed images can be used with the
        gradient of the corresponding modality in the same fasion as diff-demons
        does for mono-modality images. If the flag self.useDoubleGradient is True
        these garadients are averaged.
        '''
        self.__connectFunctions()
        samplingMask=self.fixedImageMask*self.movingImageMask
        self.samplingMask=samplingMask
        fixedQ, self.fixedQLevels, hist=self.quantize(self.fixedImage, self.quantizationLevels)
        fixedQ=np.array(fixedQ, dtype=np.int32)
        self.fixedQLevels=np.array(self.fixedQLevels)
        fixedQMeans, fixedQVariances=self.computeStats(samplingMask, self.movingImage, self.quantizationLevels, fixedQ)
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
        movingQ, self.movingQLevels, hist=self.quantize(self.movingImage, self.quantizationLevels)
        movingQ=np.array(movingQ, dtype=np.int32)
        self.movingQLevels=np.array(self.movingQLevels)
        movingQMeans, movingQVariances=self.computeStats(samplingMask, self.fixedImage, self.quantizationLevels, movingQ)        
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
                self.gradientFixed[...,i]+=grad
                i+=1

    def freeIteration(self):
        r'''
        Frees the resources allocated during initialization
        '''
        del self.samplingMask
        del self.fixedQLevels
        del self.movingQLevels
        del self.fixedQSigmaField
        del self.fixedQMeansField
        del self.movingQSigmaField
        del self.movingQMeansField
        del self.gradientMoving
        del self.gradientFixed

    def computeForward(self):
        r'''
        Computes the update displacement field to be used for registration of
        the moving image towards the fixed image
        '''
        return self.computeStep(True)

    def computeBackward(self):
        r'''
        Computes the update displacement field to be used for registration of
        the fixed image towards the moving image
        '''
        return self.computeStep(False)

    def computeGaussSeidelStep(self, forwardStep=True):
        r'''
        Minimizes the linearized energy function with respect to the regularized
        displacement field (this step does not require post-smoothing, as opposed
        to the demons step, which does not include regularization).
        To accelerate convergence we use the multi-grid Gauss-Seidel algorithm
        proposed by Bruhn and Weickert et al [1]
        [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation: 
            combining highest accuracy with real-time performance", 
            10th IEEE International Conference on Computer Vision, 2005. ICCV 2005.         
        '''
        maxInnerIter=self.parameters['maxInnerIter']
        lambdaParam=self.parameters['lambda']
        maxStepLength=self.parameters['maxStepLength']
        sh=self.fixedImage.shape if forwardStep else self.movingImage.shape
        deltaField=self.fixedQMeansField-self.movingImage if forwardStep else self.movingQMeansField - self.fixedImage
        sigmaField=self.fixedQSigmaField if forwardStep else self.movingQSigmaField
        gradient=self.gradientMoving if forwardStep else self.gradientFixed
        displacement=np.zeros(shape=(sh)+(self.dim,), dtype=np.float64)
        self.energy=self.multiResolutionIteration(self.levelsBelow, maxInnerIter, deltaField, sigmaField, gradient, None, lambdaParam, displacement)
        maxNorm=np.sqrt(np.sum(displacement**2,-1)).max()
        if maxNorm>maxStepLength:
            displacement*=maxStepLength/maxNorm
        return displacement

    def computeDemonsStep(self, forwardStep=True):
        return NotImplemented

    def getEnergy(self):
        return NotImplemented

    def useOriginalFixedImage(self, originalFixedImage):
        r'''
        EMMetric computes the object mask by thresholding the original fixed
        image
        '''
        pass

    def useOriginalMovingImage(self, originalMovingImage):
        r'''
        EMMetric computes the object mask by thresholding the original moving
        image
        '''
        pass

    def useFixedImageDynamics(self, originalFixedImage, transformation, direction):
        r'''
        EMMetric takes advantage of the image dynamics by computing the
        current fixed image mask from the originalFixedImage mask (warped
        by nearest neighbor interpolation)
        '''
        self.fixedImageMask=(originalFixedImage>0).astype(np.int32)
        if transformation==None:
            self.fixedImageMask=self.fixedImageMask
            return
        if direction==1:
            self.fixedImageMask=transformation.warp_forward_nn(self.fixedImageMask)
        else:
            self.fixedImageMask=transformation.warp_backward_nn(self.fixedImageMask)

    def useMovingImageDynamics(self, originalMovingImage, transformation, direction):
        r'''
        EMMetric takes advantage of the image dynamics by computing the
        current moving image mask from the originalMovingImage mask (warped
        by nearest neighbor interpolation)
        '''
        self.movingImageMask=(originalMovingImage>0).astype(np.int32)
        if transformation==None:
            self.movingImageMask=self.movingImageMask
            return
        if direction==1:
            self.movingImageMask=transformation.warp_forward_nn(self.movingImageMask)
        else:
            self.movingImageMask=transformation.warp_backward_nn(self.movingImageMask)

    def reportStatus(self):
        r'''
        Shows the overlaid input images
        '''
        if self.dim==2:
            plt.figure()
            rcommon.overlayImages(self.movingQMeansField, self.fixedQMeansField, False)
        else:
            fixed=self.fixedImage
            moving=self.movingImage
            sh=self.fixedQMeansField.shape
            rcommon.overlayImages(moving[:,sh[1]//2,:], fixed[:,sh[1]//2,:])
            rcommon.overlayImages(moving[sh[0]//2,:,:], fixed[sh[0]//2,:,:])
            rcommon.overlayImages(moving[:,:,sh[2]//2], fixed[:,:,sh[2]//2])    

    def getMetricName(self):
        return "EMMetric"
