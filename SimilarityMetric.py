import abc
class SimilarityMetric(object):
    '''
    A similarity metric is in charge of keeping track of the numerical value
    of the similarity (or distance) between the two given images. It also
    computes the update field for the forward and inverse
    displacement fields to be used in a gradient-based optimization algorithm.
    Note that this metric does not depend on any transformation (affine or
    non-linear), so it assumes the fixed and reference images are already warped
    '''
    __metaclass__ = abc.ABCMeta
    def __init__(self, parameters):
        defaultParameters = self.getDefaultParameters()
        for key, val in parameters.iteritems():
            if key in defaultParameters:
                defaultParameters[key] = val
            else:
                print "Warning: parameter '",key,"' unknown. Ignored."
        self.parameters = defaultParameters
        self.setFixedImage(None)
        self.setMovingImage(None)
        self.symmetric = False
        self.dim = None

    def setLevelsBelow(self, levels):
        self.levelsBelow = levels

    def setLevelsAbove(self, levels):
        self.levelsAbove = levels

    def setFixedImage(self, fixedImage):
        '''
        Sets the fixed image. The dimension the similarity metric operates on
        is defined as the dimension of the last image (fixed or moving) passed
        to it
        '''
        self.dim = len(fixedImage.shape) if fixedImage != None else 0
        self.fixedImage = fixedImage

    @abc.abstractmethod
    def getMetricName(self):
        '''
        Must return the name of the metric that specializes this generic metric
        '''
        pass

    @abc.abstractmethod
    def useFixedImageDynamics(self,
                              originalFixedImage,
                              transformation,
                              direction):
        '''
        This methods provides the metric a chance to compute any useful
        information from knowing how the current fixed image was generated
        (as the transformation of an original fixed image). This method is
        called by the optimizer just after it sets the fixed image.
        Transformation will be an instance of TransformationModel or None if
        the originalMovingImage equals self.movingImage. Direction is either 1
        (warp forward) or -1(warp backward)
        '''
        pass

    @abc.abstractmethod
    def useOriginalFixedImage(self, originalFixedImage):
        '''
        This methods provides the metric a chance to compute any useful
        information from the original moving image (to be used along with the
        sequence of movingImages during optimization, for example the binary
        mask delimiting the object of interest can be computed from the original
        image only and then warp this binary mask instead of thresholding
        at each iteration, which might cause artifacts due to interpolation)
        '''
        pass

    def setMovingImage(self, movingImage):
        '''
        Sets the moving image. The dimension the similarity metric operates on
        is defined as the dimension of the last image (fixed or moving) passed
        to it
        '''
        self.dim = len(movingImage.shape) if movingImage != None else 0
        self.movingImage = movingImage

    @abc.abstractmethod
    def useOriginalMovingImage(self, originalMovingImage):
        '''
        This methods provides the metric a chance to compute any useful
        information from the original moving image (to be used along with the
        sequence of movingImages during optimization, for example the binary
        mask delimiting the object of interest can be computed from the original
        image only and then warp this binary mask instead of thresholding
        at each iteration, which might cause artifacts due to interpolation)
        '''
        pass

    @abc.abstractmethod
    def useMovingImageDynamics(self,
                               originalMovingImage,
                               transformation,
                               direction):
        '''
        This methods provides the metric a chance to compute any useful
        information from knowing how the current fixed image was generated
        (as the transformation of an original fixed image). This method is
        called by the optimizer just after it sets the fixed image.
        Transformation will be an instance of TransformationModel or None if
        the originalMovingImage equals self.movingImage. Direction is either 1
        (warp forward) or -1(warp backward)
        '''
        pass

    @abc.abstractmethod
    def initializeIteration(self):
        '''
        This method will be called before any computeUpdate or computeInverse
        call, this gives the chance to the Metric to precompute any useful
        information for speeding up the update computations. This initialization
        was needed in ANTS because the updates are called once per voxel. In
        Python this is unpractical, though.
        '''
        return NotImplemented

    @abc.abstractmethod
    def freeIteration(self):
        '''
        This method is called by the RegistrationOptimizer after the required
        iterations have been computed (forward and/or backward) so that the
        SimilarityMetric can safely delete any data it computed as part of the
        initialization
        '''
        return NotImplemented

    @abc.abstractmethod
    def computeForward(self):
        '''
        Must return the forward update field for a gradient-based optimization
        algorithm
        '''
        return NotImplemented

    @abc.abstractmethod
    def computeBackward(self):
        '''
        Must return the inverse update field for a gradient-based optimization
        algorithm
        '''
        return NotImplemented

    @abc.abstractmethod
    def getEnergy(self):
        '''
        Must return the numeric value of the similarity between the given fixed
        and moving images
        '''
        return NotImplemented

    @abc.abstractmethod
    def getDefaultParameters(self):
        return NotImplemented

    @abc.abstractmethod
    def reportStatus(self):
        '''
        This function is called mostly for debugging purposes. The metric
        can for example show the overlaid images or print some statistics
        '''
        return NotImplemented