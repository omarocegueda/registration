import numpy as np
from TransformationModel import TransformationModel
import abc

class RegistrationOptimizer(object):
    r'''
    This abstract class defines the interface to be implemented by any
    optimization algorithm for nonlinear Registration
    '''
    @abc.abstractmethod
    def get_default_parameters(self):
        return NotImplemented

    def __init__(self, fixed=None, moving=None, affineFixed=None, affineMoving=None, similarityMetric=None, updateRule=None, parameters=None):
        defaultParameters=self.get_default_parameters()
        if parameters!=None:
            for key, val in parameters.iteritems():
                if key in defaultParameters:
                    defaultParameters[key]=val
                else:
                    print "Warning: parameter '",key,"' unknown. Ignored."
        if affineFixed!=None:
            print 'Warning: an affineFixed matrix was given as argument. This functionality has not been implemented yet.'
        self.parameters=defaultParameters
        invAffineMoving=None if affineMoving==None else np.linalg.inv(affineMoving).copy(order='C')
        self.dim=0
        self.setFixedImage(fixed)
        self.forward_model=TransformationModel(None, None, None, None)
        self.setMovingImage(moving)
        self.backward_model=TransformationModel(None, None, invAffineMoving, None)
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
        r'''
        This is the main function each especialized class derived from this must
        implement. Upon completion, the deformation field must be available from
        the forward transformation model.
        '''
        return NotImplemented

    def get_forward(self):
        return self.forward_model.forward

    def get_backward(self):
        return self.forward_model.backward
