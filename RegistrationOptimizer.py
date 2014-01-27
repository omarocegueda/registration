import numpy as np
from TransformationModel import TransformationModel
import abc

class RegistrationOptimizer(object):
    '''
    This abstract class defines the interface to be implemented by any
    optimization algorithm for nonlinear Registration
    '''
    @abc.abstractmethod
    def getDefaultParameters(self):
        return NotImplemented

    def __init__(self, fixed=None, moving=None, affineFixed=None, affineMoving=None, similarityMetric=None, updateRule=None, parameters=None):
        defaultParameters=self.getDefaultParameters()
        if parameters!=None:
            for key, val in parameters.iteritems():
                if key in defaultParameters:
                    defaultParameters[key]=val
                else:
                    print "Warning: parameter '",key,"' unknown. Ignored."
        self.parameters=defaultParameters
        invAffineMoving=None if affineMoving==None else np.linalg.inv(affineMoving).copy(order='C')
        self.dim=0
        self.setFixedImage(fixed)
        self.forwardModel=TransformationModel(None, None, None, None)
        self.setMovingImage(moving)
        self.backwardModel=TransformationModel(None, None, invAffineMoving, None)
        self.similarityMetric=similarityMetric
        self.updateRule=updateRule
        self.energyList=None

    def setSimilarityMetric(self, similarityMetric):
        self.similarityMetric=similarityMetric

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

    @abc.abstractmethod
    def optimize(self):
        return NotImplemented

    def getForward(self):
        return self.forwardModel.getForward()

    def getBackward(self):
        return self.forwardModel.getBackward()
