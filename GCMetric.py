import numpy as np
import scipy as sp
from scipy import gradient, ndimage
from dipy.align.metrics import SimilarityMetric
import grad_corr as gc
from dipy.align import floating

class GCMetric(SimilarityMetric):

    def __init__(self, dim, step_length = 0.25, sigma_diff = 3.0):
        r"""
        Similarity metric defined as the sum of the squared normalized correlation 
        of local gradients

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the image domain
        step_length : float
            the length of the maximum displacement vector of the displacement
            update displacement field at each iteration
        sigma_diff : the standard deviation of the Gaussian smoothing kernel to
            be applied to the update field at each iteration
        """
        super(GCMetric, self).__init__(dim)
        self.step_length = step_length
        self.sigma_diff = sigma_diff

    def initialize_iteration(self):
        r"""
        Precomputes the cross-correlation factors
        """
        sigma=0.001
        self.factors = gc.precompute_gc_factors_2d(self.static_image, self.moving_image)
        i = 0
        while i < 6:
            self.factors[..., i] = ndimage.filters.gaussian_filter(self.factors[..., i],
                                                                   sigma)
            i+=1
        self.factors = np.array(self.factors)
        self.gradient_moving = np.empty(
            shape = (self.moving_image.shape)+(self.dim,), dtype = floating)
        i = 0
        for grad in sp.gradient(self.moving_image):
            self.gradient_moving[..., i] = grad
            i += 1
        self.gradient_static = np.empty(
            shape = (self.static_image.shape)+(self.dim,), dtype = floating)
        i = 0
        for grad in sp.gradient(self.static_image):
            self.gradient_static[..., i] = grad
            i += 1

    def free_iteration(self):
        r"""
        Frees the resources allocated during initialization
        """
        del self.factors
        del self.gradient_moving
        del self.gradient_static
    
    def compute_forward(self):
        r"""
        Computes the update displacement field to be used for registration of
        the moving image towards the static image
        """
        displacement, self.energy=gc.compute_gc_forward_step_2d(
            self.gradient_static, self.gradient_moving, self.factors)
        displacement=np.array(displacement)
        displacement[..., 0] = ndimage.filters.gaussian_filter(displacement[..., 0],
                                                               self.sigma_diff)
        displacement[..., 1] = ndimage.filters.gaussian_filter(displacement[..., 1],
                                                                self.sigma_diff)
        max_norm = np.sqrt(np.sum(displacement**2, -1)).max()
        displacement *= self.step_length/max_norm
        return displacement

    def compute_backward(self):
        r"""
        Computes the update displacement field to be used for registration of
        the static image towards the moving image
        """
        displacement, energy=gc.compute_gc_backward_step_2d(self.gradient_static, self.gradient_moving, self.factors)
        displacement=np.array(displacement)
        displacement[..., 0] = ndimage.filters.gaussian_filter(displacement[..., 0],
                                                               self.sigma_diff)
        displacement[..., 1] = ndimage.filters.gaussian_filter(displacement[..., 1],
                                                                self.sigma_diff)
        max_norm = np.sqrt(np.sum(displacement**2, -1)).max()
        displacement *= self.step_length/max_norm
        return displacement


    def get_energy(self):
        r"""
        Returns the Cross Correlation (data term) energy computed at the largest
        iteration
        """
        return self.energy

    def use_static_image_dynamics(self, original_static_image, transformation):
        r"""
        CCMetric takes advantage of the image dynamics by computing the
        current static image mask from the originalstaticImage mask (warped
        by nearest neighbor interpolation)

        Parameters
        ----------
        original_static_image : array, shape (R, C) or (S, R, C)
            the original image from which the current static image was generated
        transformation : DiffeomorphicMap object
            the transformation that was applied to original image to generate
            the current static image
        """
        self.static_image_mask = (original_static_image>0).astype(np.int32)
        if transformation == None:
            return
        self.static_image_mask = transformation.transform(self.static_image_mask,'nn')

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r"""
        CCMetric takes advantage of the image dynamics by computing the
        current moving image mask from the originalMovingImage mask (warped
        by nearest neighbor interpolation)

        Parameters
        ----------
        original_moving_image : array, shape (R, C) or (S, R, C)
            the original image from which the current moving image was generated
        transformation : DiffeomorphicMap object
            the transformation that was applied to original image to generate
            the current moving image
        """
        self.moving_image_mask = (original_moving_image>0).astype(np.int32)
        if transformation == None:
            return
        self.moving_image_mask = transformation.transform(self.moving_image_mask, 'nn')

def test_gc_2d(metric_type = 'gc'):
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration
    '''
    import numpy as np
    from numpy.testing import (assert_equal,
                               assert_array_equal,
                               assert_array_almost_equal)
    import matplotlib.pyplot as plt
    import dipy.align.imwarp as imwarp
    import dipy.align.metrics as metrics 
    import dipy.align.vector_fields as vfu
    from dipy.data import get_data
    from dipy.align import floating
    import nibabel as nib
    import registrationCommon as rcommon

    fname_moving = get_data('reg_o')
    fname_static = get_data('reg_c')

    moving = plt.imread(fname_moving)
    static = plt.imread(fname_static)
    moving = moving[:, :, 0].astype(floating)
    static = static[:, :, 0].astype(floating)
    moving = np.array(moving, dtype = floating)
    static = np.array(static, dtype = floating)
    moving = (moving-moving.min())/(moving.max() - moving.min())
    static = (static-static.min())/(static.max() - static.min())
    #Configure the metric
    if metric_type == 'gc':
        smooth = 4
        step_length = 0.25
        similarity_metric = GCMetric(2, step_length, smooth) 
    else:
        smooth = 4
        inner_iter =5
        step_length = 0.25
        step_type = 0
        similarity_metric = metrics.SSDMetric(2, smooth, inner_iter, step_length, step_type) 
    #Configure and run the Optimizer
    opt_iter = [50, 100, 100, 100,100]
    opt_tol = -1e4
    inv_iter = 40
    inv_tol = 1e-3
    registration_optimizer = imwarp.SymmetricDiffeomorphicRegistration(
        similarity_metric, opt_iter, opt_tol, inv_iter, inv_tol)
    registration_optimizer.verbosity = 2
    mapping = registration_optimizer.optimize(static, moving, None)
    rcommon.plotDiffeomorphism(mapping.forward, mapping.backward, np.zeros_like(mapping.forward), '')
    img0 = static
    img1 = mapping.transform(moving)
    rcommon.overlayImages(img0, img1, True)

if __name__=="__main__":
    test_gc_2d()
