import abc
class SimilarityMetric(object):
    '''
    A similarity metric is in charge of keeping track of the numerical value
    of the similarity (or distance) between the two given images. It also 
    computes the update field ("negative gradient") for the forward and inverse 
    displacement fields to be used in a gradient-based optimization algorithm.
    Note that this metric does not depend on any transformation (affine or 
    non-linear), so it assumes the fixed and reference images are already warped
    '''
    __metaclass__ = abc.ABCMeta
    def __init__(self, fixedImage=None, movingImage=None):
        self.setFixedImage(fixedImage)
        self.setMovingImage(movingImage)        

    def setFixedImage(self, fixedImage):
        self.dim=len(fixedImage.shape) if fixedImage!=None else 0
        self.fixedImage=fixedImage

    def setMovingImage(self, movingImage):
        self.dim=len(movingImage.shape) if movingImage!=None else 0
        self.movingImage=movingImage

    @abc.abstractmethod
    def initializeIteration(self):
        '''
        This method will be called before any computeUpdate or computeInverse call,
        this gives the chance to the Metric to precompute any useful information
        for speeding up the update computations. This initialization was needed
        in ANTS because the updates are called once per voxel. In Python this is
        unpractical, though.
        '''
        return NotImplemented

    @abc.abstractmethod
    def computeForward(self):
        '''
        Must return the forward update field for a gradient-based optimization algorithm
        '''
        return NotImplemented

    @abc.abstractmethod
    def computeBackward(self):
        '''
        Must return the inverse update field for a gradient-based optimization algorithm
        '''
        return NotImplemented

    @abc.abstractmethod
    def getEnergy(self):
        '''
        Must return the numeric value of the similarity between the given fixed and
        moving images
        '''
        return NotImplemented
