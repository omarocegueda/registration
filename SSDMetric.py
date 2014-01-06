import numpy as np
import scipy as sp
import tensorFieldUtils as tf
from SimilarityMetric import SimilarityMetric
class SSDMetric(SimilarityMetric):
    GAUSS_SEIDEL_STEP=0
    DEMONS_STEP=1
    def __init__(self, parameters):
        super(SSDMetric, self).__init__()
        defaultParameters={'lambda':1.0, 'maxInnerIter':200, 'innerTolerance':1e-4, 
                           'scale':1, 'maxStepLength':1.0, 'sigmaDiff':2.0}
        for key, val in parameters.iteritems():
            if key in defaultParameters:
                defaultParameters[key]=val
            else:
                print "Warning: unknown parameter '",key,"' unknown"
        parameters=defaultParameters
        self.parameters=parameters
        self.updateType=SSDMetric.GAUSS_SEIDEL_STEP

    def initializeIteration(self):
        self.sigmaField=np.ones_like(self.movingImage, dtype=np.float64)
        self.deltaField=self.fixedImage-self.movingImage
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
        if self.updateType==SSDMetric.GAUSS_SEIDEL_STEP:
            return self.computeGaussSeidelStep(True)
        elif self.updateType==SSDMetric.DEMONS_STEP:
            return self.computeDemonsStep(True)
        return None

    def computeBackward(self):
        if self.updateType==SSDMetric.GAUSS_SEIDEL_STEP:
            return self.computeGaussSeidelStep(False)
        elif self.updateType==SSDMetric.DEMONS_STEP:
            return self.computeDemonsStep(False)
        return None

    def computeGaussSeidelStep(self, forwardStep=True):
        maxInnerIter=self.parameters['maxInnerIter']
        tolerance=self.parameters['innerTolerance']
        lambdaParam=self.parameters['lambda']
        sh=self.fixedImage.shape if forwardStep else self.movingImage.shape
        deltaField=self.deltaField if forwardStep else self.deltaField*-1.0
        gradient=self.gradientMoving+self.gradientFixed if forwardStep else self.gradientFixed
        displacement=np.zeros(shape=(sh)+(self.dim,), dtype=np.float64)
        error=1+tolerance
        innerIter=0
        if self.dim==2:
            while((error>tolerance)and(innerIter<maxInnerIter)):
                innerIter+=1
                error=tf.iterateDisplacementField2DCYTHON(deltaField, None, gradient,  lambdaParam, displacement, None)
        else:
            while((error>tolerance)and(innerIter<maxInnerIter)):
                innerIter+=1
                error=tf.iterateDisplacementField3DCYTHON(deltaField, None, gradient,  lambdaParam, displacement, None)
        return displacement

    def computeDemonsStep(self, forwardStep=True):
        sigmaDiff=self.parameters['sigmaDiff']
        maxStepLength=self.parameters['maxStepLength']
        scale=self.parameters['scale']
        deltaField=self.deltaField if forwardStep else self.deltaField*-1.0
        gradient=self.gradientMoving if forwardStep else self.gradientFixed
        if self.dim==2:
            forward=tf.compute_demons_step2D(deltaField, gradient, maxStepLength, scale)
            forward[...,0]=sp.ndimage.filters.gaussian_filter(forward[...,0], sigmaDiff)
            forward[...,1]=sp.ndimage.filters.gaussian_filter(forward[...,1], sigmaDiff)
        else:
            return NotImplemented
        return forward

    def setUpdateType(self, updateType):
        self.updateType=updateType

    def getEnergy(self):
        pass
