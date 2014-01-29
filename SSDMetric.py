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
    GAUSS_SEIDEL_STEP = 0
    DEMONS_STEP = 1
    def get_default_parameters(self):
        return {'lambda':1.0, 'max_inner_iter':5, 'scale':1,
                'max_step_length':0.25, 'sigma_diff':3.0, 'step_type':0}

    def __init__(self, parameters):
        super(SSDMetric, self).__init__(parameters)
        self.step_type = self.parameters['step_type']
        self.levels_below = 0

    def initialize_iteration(self):
        r'''
        Precomputes the gradient of the input images to be used in the
        computation of the forward and backward steps.
        '''
        self.gradient_moving = np.empty(
            shape = (self.moving_image.shape)+(self.dim,), dtype = np.float64)
        i = 0
        for grad in sp.gradient(self.moving_image):
            self.gradient_moving[...,i] = grad
            i+= 1
        i = 0
        self.gradient_fixed = np.empty(
            shape = (self.fixed_image.shape)+(self.dim,), dtype = np.float64)
        for grad in sp.gradient(self.fixed_image):
            self.gradient_fixed[...,i] = grad
            i+= 1

    def compute_forward(self):
        r'''
        Computes the update displacement field to be used for registration of
        the moving image towards the fixed image
        '''
        if self.step_type == SSDMetric.GAUSS_SEIDEL_STEP:
            return self.compute_gauss_seidel_step(True)
        elif self.step_type == SSDMetric.DEMONS_STEP:
            return self.compute_demons_step(True)
        return None

    def compute_backward(self):
        r'''
        Computes the update displacement field to be used for registration of
        the fixed image towards the moving image
        '''
        if self.step_type == SSDMetric.GAUSS_SEIDEL_STEP:
            return self.compute_gauss_seidel_step(False)
        elif self.step_type == SSDMetric.DEMONS_STEP:
            return self.compute_demons_step(False)
        return None

    def compute_gauss_seidel_step(self, forwardStep = True):
        r'''
        Minimizes the linearized energy function defined by the sum of squared
        differences of corresponding pixels of the input images with respect
        to the displacement field.
        '''
        maxInnerIter = self.parameters['maxInnerIter']
        lambdaParam = self.parameters['lambda']
        maxStepLength = self.parameters['maxStepLength']
        sh = self.fixed_image.shape if forwardStep else self.moving_image.shape
        if forwardStep:
            deltaField = self.fixed_image-self.moving_image
        else:
            deltaField = self.moving_image - self.fixed_image
        gradient = self.gradient_moving+self.gradient_fixed
        displacement = np.zeros(shape = (sh)+(self.dim,), dtype = np.float64)
        if self.dim == 2:
            displacement = wCycle2D(self.levels_below, maxInnerIter, deltaField,
                                    None, gradient, None, lambdaParam,
                                    displacement)
        else:
            displacement = vCycle3D(self.levels_below, maxInnerIter, deltaField,
                                    None, gradient, None, lambdaParam,
                                    displacement)
        maxNorm = np.sqrt(np.sum(displacement**2, -1)).max()
        if maxNorm>maxStepLength:
            displacement*= maxStepLength/maxNorm
        return displacement

    def compute_demons_step(self, forwardStep = True):
        r'''
        Computes the demons step proposed by Vercauteren et al.[1] for the SSD
        metric.
        [1] Tom Vercauteren, Xavier Pennec, Aymeric Perchant, Nicholas Ayache,
            "Diffeomorphic Demons: Efficient Non-parametric Image Registration",
            Neuroimage 2009
        '''
        sigmaDiff = self.parameters['sigmaDiff']
        maxStepLength = self.parameters['maxStepLength']
        scale = self.parameters['scale']
        if forwardStep:
            deltaField = self.fixed_image-self.moving_image
        else:
            deltaField = self.moving_image - self.fixed_image
        gradient = self.gradient_moving+self.gradient_fixed
        if self.dim == 2:
            forward = tf.compute_demons_step2D(deltaField, gradient,
                                               maxStepLength, scale)
            forward[...,0] = sp.ndimage.filters.gaussian_filter(forward[...,0],
                                                                sigmaDiff)
            forward[...,1] = sp.ndimage.filters.gaussian_filter(forward[...,1],
                                                                sigmaDiff)
        else:
            forward = tf.compute_demons_step2D(deltaField, gradient,
                                               maxStepLength, scale)
            forward[...,0] = sp.ndimage.filters.gaussian_filter(forward[...,0],
                                                                sigmaDiff)
            forward[...,1] = sp.ndimage.filters.gaussian_filter(forward[...,1],
                                                                sigmaDiff)
            forward[...,2] = sp.ndimage.filters.gaussian_filter(forward[...,2],
                                                                sigmaDiff)
        return forward

    def get_energy(self):
        return NotImplemented

    def use_original_fixed_image(self, originalfixed_image):
        r'''
        SSDMetric does not take advantage of the original fixed image, just pass
        '''
        pass

    def use_original_moving_image(self, originalMovingImage):
        r'''
        SSDMetric does not take advantage of the original moving image just pass
        '''
        pass

    def use_fixed_image_dynamics(self, originalfixed_image, transformation,
                                 direction):
        r'''
        SSDMetric does not take advantage of the image dynamics, just pass
        '''
        pass

    def use_moving_image_dynamics(self, originalMovingImage, transformation,
                                  direction):
        r'''
        SSDMetric does not take advantage of the image dynamics, just pass
        '''
        pass

    def report_status(self):
        plt.figure()
        rcommon.overlayImages(self.moving_image, self.fixed_image, False)

    def get_metric_name(self):
        return "SSDMetric"
#######################Multigrid algorithms for SSD-like metrics#############

printEnergy = False
def singleCycle2D(n, k, deltaField, sigmaField, gradientField, lambdaParam,
                  displacement, depth = 0):
    r'''
    One-pass multi-resolution Gauss-Seidel solver: solves the SSD-like linear
    system starting at the coarcer resolution and then refining at the finer
    resolution.
    '''
    iterFactor = 1
    if n == 0:
        for i in range(k*iterFactor):
            error = tf.iterate_residual_displacement_field_SSD2D(deltaField,
                                                                 sigmaField,
                                                                 gradientField,
                                                                 None,
                                                                 lambdaParam,
                                                                 displacement)
            if printEnergy and depth == 0:
                energy = tf.compute_energy_SSD2D(deltaField,
                                                 sigmaField,
                                                 gradientField,
                                                 lambdaParam,
                                                 displacement)
                print 'Energy after top-level iter',i+1,' [unique]:',energy
        return error
    #solve at coarcer grid
    subSigmaField = None
    if sigmaField!= None:
        subSigmaField = tf.downsample_scalar_field(sigmaField)
    subDeltaField = tf.downsample_scalar_field(deltaField)
    subGradientField = np.array(tf.downsample_displacement_field(gradientField))
    sh = np.array(displacement.shape).astype(np.int32)
    subDisplacement = np.zeros(
        shape = ((sh[0]+1)//2, (sh[1]+1)//2, 2 ), dtype = np.float64)
    subLambdaParam = lambdaParam*0.25
    singleCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField,
                  subLambdaParam, subDisplacement, depth+1)
    displacement+= np.array(tf.upsample_displacement_field(subDisplacement, sh))
    if printEnergy and depth == 0:
        energy = tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD2D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             None,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD2D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    try:
        energy
    except NameError:
        energy = tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
    return energy

def vCycle2D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam,
             displacement, depth = 0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by first
    filtering (GS-iterate) the current level, then solves for the residual
    at a coarcer resolution andfinally refines the solution at the current
    resolution. This scheme corresponds to the V-cycle proposed by Bruhn and
    Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation:
            combining highest accuracy with real-time performance",
            10th IEEE International Conference on Computer Vision, 2005.
            ICCV 2005.
    '''
    iterFactor = 1
    #presmoothing
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD2D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             target,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD2D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    if n == 0:
        return error
    #solve at coarcer grid
    residual = None
    residual = tf.compute_residual_displacement_field_SSD2D(deltaField,
                                                            sigmaField,
                                                            gradientField,
                                                            target,
                                                            lambdaParam,
                                                            displacement,
                                                            residual)
    subResidual = np.array(tf.downsample_displacement_field(residual))
    del residual
    subSigmaField = None
    if sigmaField!= None:
        subSigmaField = tf.downsample_scalar_field(sigmaField)
    subDeltaField = tf.downsample_scalar_field(deltaField)
    subGradientField = np.array(tf.downsample_displacement_field(gradientField))
    sh = np.array(displacement.shape).astype(np.int32)
    #subDisplacement = np.array(tf.downsample_displacement_field(displacement))
    subDisplacement = np.zeros(shape = ((sh[0]+1)//2, (sh[1]+1)//2, 2 ),
                               dtype = np.float64)
    subLambdaParam = lambdaParam*0.25
    vCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField,
             subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+= np.array(tf.upsample_displacement_field(subDisplacement, sh))
    if printEnergy and depth == 0:
        energy = tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD2D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             target,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD2D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    try:
        energy
    except NameError:
        energy = tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
    return energy

def wCycle2D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam,
             displacement, depth = 0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by performing
    two v-cycles at each resolution, which corresponds to the w-cycle scheme
    proposed by Bruhn and Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation:
            combining highest accuracy with real-time performance",
            10th IEEE International Conference on Computer Vision, 2005.
            ICCV 2005.
    '''
    iterFactor = 1
    #presmoothing
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD2D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             target,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD2D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [first]:',energy
    if n == 0:
        return error
    residual = tf.compute_residual_displacement_field_SSD2D(deltaField,
                                                            sigmaField,
                                                            gradientField,
                                                            target,
                                                            lambdaParam,
                                                            displacement,
                                                            None)
    subResidual = np.array(tf.downsample_displacement_field(residual))
    del residual
    #solve at coarcer grid
    subSigmaField = None
    if sigmaField!= None:
        subSigmaField = tf.downsample_scalar_field(sigmaField)
    subDeltaField = tf.downsample_scalar_field(deltaField)
    subGradientField = np.array(tf.downsample_displacement_field(gradientField))
    sh = np.array(displacement.shape).astype(np.int32)
    #subDisplacement = np.array(tf.downsample_displacement_field(displacement))
    subDisplacement = np.zeros(shape = ((sh[0]+1)//2, (sh[1]+1)//2, 2 ),
                               dtype = np.float64)
    subLambdaParam = lambdaParam*0.25
    wCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField,
             subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+= np.array(tf.upsample_displacement_field(subDisplacement, sh))
    if printEnergy and depth == 0:
        energy = tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
        print 'Energy after low-res iteration[first]:',energy
    #post-smoothing (second smoothing)
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD2D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             target,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD2D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [second]:',energy
    residual = tf.compute_residual_displacement_field_SSD2D(deltaField,
                                                            sigmaField,
                                                            gradientField,
                                                            target,
                                                            lambdaParam,
                                                            displacement,
                                                            None)
    subResidual = np.array(tf.downsample_displacement_field(residual))
    del residual
    #subDisplacement = np.array(tf.downsample_displacement_field(displacement))
    subDisplacement = np.zeros(shape = ((sh[0]+1)//2, (sh[1]+1)//2, 2 ),
                               dtype = np.float64)
    wCycle2D(n-1, k, subDeltaField, subSigmaField, subGradientField,
             subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+= np.array(tf.upsample_displacement_field(subDisplacement, sh))
    if printEnergy and depth == 0:
        energy = tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
        print 'Energy after low-res iteration[second]:',energy
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD2D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             target,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD2D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [third]:',energy
    try:
        energy
    except NameError:
        energy = tf.compute_energy_SSD2D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
    return energy

def singleCycle3D(n, k, deltaField, sigmaField, gradientField, lambdaParam, displacement, depth = 0):
    r'''
    One-pass multi-resolution Gauss-Seidel solver: solves the SSD-like linear
    system starting at the coarcer resolution and then refining at the finer
    resolution.
    '''
    iterFactor = 1
    if n == 0:
        for i in range(k*iterFactor):
            error = tf.iterate_residual_displacement_field_SSD3D(deltaField,
                                                                 sigmaField,
                                                                 gradientField,
                                                                 None,
                                                                 lambdaParam,
                                                                 displacement)
            if printEnergy and depth == 0:
                energy = tf.compute_energy_SSD3D(deltaField,
                                                 sigmaField,
                                                 gradientField,
                                                 lambdaParam,
                                                 displacement)
                print 'Energy after top-level iter',i+1,' [unique]:',energy
        return error
    #solve at coarcer grid
    subSigmaField = None
    if sigmaField!= None:
        subSigmaField = tf.downsample_scalar_field3D(sigmaField)
    subDeltaField = tf.downsample_scalar_field3D(deltaField)
    subGradientField = np.array(
        tf.downsample_displacement_field3D(gradientField))
    sh = np.array(displacement.shape).astype(np.int32)
    subDisplacement = np.zeros(
        shape = ((sh[0]+1)//2, (sh[1]+1)//2, (sh[2]+1)//2, 3 ),
        dtype = np.float64)
    subLambdaParam = lambdaParam*0.25
    singleCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField,
                  subLambdaParam, subDisplacement, depth+1)
    displacement+= np.array(
        tf.upsample_displacement_field3D(subDisplacement, sh))
    if printEnergy and depth == 0:
        energy = tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD3D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             None,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD3D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    try:
        energy
    except NameError:
        energy = tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
    return energy

def vCycle3D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam,
             displacement, depth = 0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by first
    filtering (GS-iterate) the current level, then solves for the residual
    at a coarcer resolution andfinally refines the solution at the current resolution.
    This scheme corresponds to the V-cycle proposed by Bruhn and Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation:
            combining highest accuracy with real-time performance",
            10th IEEE International Conference on Computer Vision, 2005. ICCV 2005.
    '''
    iterFactor = 1
    #presmoothing
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD3D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             target,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD3D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    if n == 0:
        return error
    #solve at coarcer grid
    residual = tf.compute_residual_displacement_field_SSD3D(deltaField,
                                                            sigmaField,
                                                            gradientField,
                                                            target,
                                                            lambdaParam,
                                                            displacement,
                                                            None)
    subResidual = np.array(tf.downsample_displacement_field3D(residual))
    del residual
    subSigmaField = None
    if sigmaField!= None:
        subSigmaField = tf.downsample_scalar_field3D(sigmaField)
    subDeltaField = tf.downsample_scalar_field3D(deltaField)
    subGradientField = np.array(
        tf.downsample_displacement_field3D(gradientField))
    sh = np.array(displacement.shape).astype(np.int32)
    subDisplacement = np.zeros(
        shape = ((sh[0]+1)//2, (sh[1]+1)//2, (sh[2]+1)//2, 3 ),
        dtype = np.float64)
    subLambdaParam = lambdaParam*0.25
    vCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField,
             subResidual, subLambdaParam, subDisplacement, depth+1)
    del subDeltaField
    del subSigmaField
    del subGradientField
    del subResidual
    tf.accumulate_upsample_displacement_field3D(subDisplacement, displacement)
    del subDisplacement
    if printEnergy and depth == 0:
        energy = tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
        print 'Energy after low-res iteration:',energy
    #post-smoothing
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD3D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             target,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD3D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [unique]:',energy
    try:
        energy
    except NameError:
        energy = tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
    return energy

def wCycle3D(n, k, deltaField, sigmaField, gradientField, target, lambdaParam,
             displacement, depth = 0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by performing
    two v-cycles at each resolution, which corresponds to the w-cycle scheme
    proposed by Bruhn and Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation:
            combining highest accuracy with real-time performance",
            10th IEEE International Conference on Computer Vision, 2005. ICCV 2005.
    '''
    iterFactor = 1
    #presmoothing
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD3D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             target,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD3D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [first]:',energy
    if n == 0:
        return error
    residual = tf.compute_residual_displacement_field_SSD3D(deltaField,
                                                            sigmaField,
                                                            gradientField,
                                                            target,
                                                            lambdaParam,
                                                            displacement,
                                                            None)
    subResidual = np.array(tf.downsample_displacement_field3D(residual))
    del residual
    #solve at coarcer grid
    subSigmaField = None
    if sigmaField != None:
        subSigmaField = tf.downsample_scalar_field3D(sigmaField)
    subDeltaField = tf.downsample_scalar_field3D(deltaField)
    subGradientField = np.array(
        tf.downsample_displacement_field3D(gradientField))
    sh = np.array(displacement.shape).astype(np.int32)
    subDisplacement = np.zeros(
        shape = ((sh[0]+1)//2, (sh[1]+1)//2, (sh[2]+1)//2, 3 ),
        dtype = np.float64)
    subLambdaParam = lambdaParam*0.25
    wCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField,
             subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+= np.array(
        tf.upsample_displacement_field3D(subDisplacement, sh))
    if printEnergy and depth == 0:
        energy = tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
        print 'Energy after low-res iteration[first]:',energy
    #post-smoothing (second smoothing)
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD3D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             target,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD3D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [second]:',energy
    residual = tf.compute_residual_displacement_field_SSD3D(deltaField,
                                                            sigmaField,
                                                            gradientField,
                                                            target,
                                                            lambdaParam,
                                                            displacement,
                                                            None)
    subResidual = np.array(tf.downsample_displacement_field3D(residual))
    del residual
    subDisplacement[...] = 0
    wCycle3D(n-1, k, subDeltaField, subSigmaField, subGradientField,
             subResidual, subLambdaParam, subDisplacement, depth+1)
    displacement+= np.array(
        tf.upsample_displacement_field3D(subDisplacement, sh))
    if printEnergy and depth == 0:
        energy = tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
        print 'Energy after low-res iteration[second]:',energy
    for i in range(k*iterFactor):
        error = tf.iterate_residual_displacement_field_SSD3D(deltaField,
                                                             sigmaField,
                                                             gradientField,
                                                             target,
                                                             lambdaParam,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = tf.compute_energy_SSD3D(deltaField,
                                             sigmaField,
                                             gradientField,
                                             lambdaParam,
                                             displacement)
            print 'Energy after top-level iter',i+1,' [third]:',energy
    try:
        energy
    except NameError:
        energy = tf.compute_energy_SSD3D(deltaField, sigmaField, gradientField,
                                         lambdaParam, displacement)
    return energy
