import abc
import numpy as np
import tensorFieldUtils as tf
class UpdateRule(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        pass

    @abc.abstractmethod
    def update(newDisplacement, currentDisplacement):
        '''
        Must return the updated displacement field and the mean norm of the 
        difference between the displacements before and after the update
        '''
        return NotImplemented

class Addition(UpdateRule):
    def __init__(self):
        pass
    @staticmethod
    def update(newDisplacement, currentDisplacement):
        meanNorm=np.sqrt(np.sum(newDisplacement**2,-1)).mean()
        updated=currentDisplacement+newDisplacement
        return updated, meanNorm

class Composition(UpdateRule):
    def __init__(self):
        pass
    @staticmethod
    def update(newDisplacement, currentDisplacement):
        updated, stats=tf.compose_vector_fields(newDisplacement, currentDisplacement)
        return updated, stats[0]

class ProjectedComposition(UpdateRule):
    def __init__(self):
        pass
    @staticmethod
    def update(newDisplacement, currentDisplacement):
        expd, invexpd=tf.vector_field_exponential(newDisplacement, True)
        updated, stats=tf.compose_vector_fields(expd, currentDisplacement)
        return updated, stats[0]
