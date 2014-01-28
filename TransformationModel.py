'''
Definition of the TransformationModel class, which is the composition of
an affine pre-aligning transformation followed by a nonlinear transformation
followed by an affine post-multiplication.
'''
import numpy as np
import tensorFieldUtils as tf
import numpy.linalg as linalg

def scale_affine(affine, factor):
    r'''
    Multiplies the translation part of the affine transformation by a factor
    to be used with upsampled/downsampled images (if the affine transformation)
    corresponds to an Image I and we need to apply the corresponding
    transformation to a downsampled version J of I, then the affine matrix
    is the same as for I but the translation is scaled.
    '''
    scaled_affine = affine.copy()
    domain_dimension = affine.shape[1]-1
    scaled_affine[:domain_dimension, domain_dimension] *= factor
    return scaled_affine

class TransformationModel(object):
    '''
    This class maps points between two spaces: "reference space" and "target
    space"
    Forward: maps target to reference, y=affine_post*forward(affine_pre*x)
    Backward: maps reference to target,
    x = affine_pre^{-1}*backward(affine_post^{-1}*y)
    '''
    def __init__(self,
                 forward = None,
                 backward = None,
                 affine_pre = None,
                 affine_post = None):
        self.dim = None
        self.set_forward(forward)
        self.set_backward(backward)
        self.set_affine_pre(affine_pre)
        self.set_affine_post(affine_post)

    def set_affine_pre(self, affine_pre):
        r'''
        Establishes the pre-multiplication affine matrix of this
        transformation, computes its inverse and adjusts the dimension of
        the transformation's domain accordingly
        '''
        if affine_pre != None:
            self.dim = affine_pre.shape[1]-1
            self.affine_pre_inv = linalg.inv(affine_pre).copy(order='C')
        else:
            self.affine_pre_inv = None
        self.affine_pre = affine_pre

    def set_affine_post(self, affine_post):
        r'''
        Establishes the post-multiplication affine matrix of this
        transformation, computes its inverse and adjusts the dimension of
        the transformation's domain accordingly
        '''
        if affine_post != None:
            self.dim = affine_post.shape[1]-1
            self.affine_post_inv = linalg.inv(affine_post).copy(order='C')
        else:
            self.affine_post_inv = None
        self.affine_post = affine_post

    def set_forward(self, forward):
        r'''
        Establishes the forward non-linear displacement field and adjusts
        the dimension of the transformation's domain accordingly
        '''
        if forward != None:
            self.dim = len(forward.shape)-1
        self.forward = forward

    def set_backward(self, backward):
        r'''
        Establishes the backward non-linear displacement field and adjusts
        the dimension of the transformation's domain accordingly
        '''
        if backward != None:
            self.dim = len(backward.shape)-1
        self.backward = backward

    def warp_forward(self, image):
        r'''
        Applies this transformation in the forward direction to the given image
        using tri-linear interpolation
        '''
        if len(image.shape) == 3:
            if image.dtype is np.dtype('int32'):
                warped = np.array(
                    tf.warp_discrete_volumeNN(
                        image, self.forward, self.affine_pre, self.affine_post))
            elif image.dtype is np.dtype('float64'):
                warped = np.array(
                    tf.warp_volume(
                        image, self.forward, self.affine_pre, self.affine_post))
        else:
            if image.dtype is np.dtype('int32'):
                warped = np.array(
                    tf.warp_discrete_imageNN(
                        image, self.forward, self.affine_pre, self.affine_post))
            elif image.dtype is np.dtype('float64'):
                warped = np.array(
                    tf.warp_image(
                        image, self.forward, self.affine_pre, self.affine_post))
        return warped

    def warp_backward(self, image):
        r'''
        Applies this transformation in the backward direction to the given
        image using tri-linear interpolation
        '''
        if len(image.shape) == 3:
            if image.dtype is np.dtype('int32'):
                warped = np.array(
                    tf.warp_discrete_volumeNN(
                        image, self.backward, self.affine_post_inv,
                        self.affine_pre_inv))
            elif image.dtype is np.dtype('float64'):
                warped = np.array(
                    tf.warp_volume(
                        image, self.backward, self.affine_post_inv,
                        self.affine_pre_inv))
        else:
            if image.dtype is np.dtype('int32'):
                warped = np.array(
                    tf.warp_discrete_imageNN(
                        image, self.backward, self.affine_post_inv,
                        self.affine_pre_inv))
            elif image.dtype is np.dtype('float64'):
                warped = np.array(
                    tf.warp_image(
                        image, self.backward, self.affine_post_inv,
                        self.affine_pre_inv))
        return warped

    def warp_forward_nn(self, image):
        r'''
        Applies this transformation in the forward direction to the given image
        using nearest-neighbor interpolation
        '''
        if len(image.shape) == 3:
            if image.dtype is np.dtype('int32'):
                warped = np.array(
                    tf.warp_discrete_volumeNN(
                        image, self.forward, self.affine_pre, self.affine_post))
            elif image.dtype is np.dtype('float64'):
                warped = np.array(
                    tf.warp_volumeNN(
                        image, self.forward, self.affine_pre, self.affine_post))
        else:
            if image.dtype is np.dtype('int32'):
                warped = np.array(
                    tf.warp_discrete_imageNN(
                        image, self.forward, self.affine_pre, self.affine_post))
            elif image.dtype is np.dtype('float64'):
                warped = np.array(
                    tf.warp_imageNN(
                        image, self.forward, self.affine_pre, self.affine_post))
        return warped

    def warp_backward_nn(self, image):
        r'''
        Applies this transformation in the backward direction to the given
        image using nearest-neighbor interpolation
        '''
        if len(image.shape) == 3:
            if image.dtype is np.dtype('int32'):
                warped = np.array(
                    tf.warp_discrete_volumeNN(
                        image, self.backward, self.affine_post_inv,
                        self.affine_pre_inv))
            elif image.dtype is np.dtype('float64'):
                warped = np.array(
                    tf.warp_volumeNN(
                        image, self.backward, self.affine_post_inv,
                        self.affine_pre_inv))
        else:
            if image.dtype is np.dtype('int32'):
                warped = np.array(
                    tf.warp_discrete_imageNN(
                        image, self.backward, self.affine_post_inv,
                        self.affine_pre_inv))
            elif image.dtype is np.dtype('float64'):
                warped = np.array(
                    tf.warp_imageNN(
                        image, self.backward, self.affine_post_inv,
                            self.affine_pre_inv))
        return warped

    def scale_affines(self, factor):
        r'''
        Scales the pre- and post-multiplication affine matrices to be used
        with a scaled domain. It updates the inverses as well.
        '''
        if self.affine_pre != None:
            self.affine_pre = scale_affine(self.affine_pre, factor)
            self.affine_pre_inv = linalg.inv(self.affine_pre).copy(order='C')
        if self.affine_post != None:
            self.affine_post = scale_affine(self.affine_post, factor)
            self.affine_post_inv = linalg.inv(self.affine_post).copy(order='C')

    def upsample(self, new_domain_forward, new_domain_backward):
        r'''
        Upsamples the displacement fields and scales the affine
        pre- and post-multiplication affine matrices by a factor of 2. The
        final outcome is that this transformation can be used in an upsampled
        domain.
        '''
        if self.dim == 2:
            if self.forward != None:
                self.forward = 2*np.array(
                    tf.upsample_displacement_field(
                        self.forward,
                        np.array(new_domain_forward).astype(np.int32)))
            if self.backward != None:
                self.backward = 2*np.array(
                    tf.upsample_displacement_field(
                        self.backward,
                        np.array(new_domain_backward).astype(np.int32)))
        else:
            if self.forward != None:
                self.forward = 2*np.array(
                    tf.upsample_displacement_field3D(
                        self.forward,
                        np.array(new_domain_forward).astype(np.int32)))
            if self.backward != None:
                self.backward = 2*np.array(
                    tf.upsample_displacement_field3D(
                        self.backward,
                        np.array(new_domain_backward).astype(np.int32)))
        self.scale_affines(2.0)


    def compute_inversion_error(self):
        r'''
        Returns the inversion error of the displacement fields
        TO-DO: the inversion error should take into account the affine
        transformations as well.
        '''
        if self.dim == 2:
            residual, stats = tf.compose_vector_fields(self.forward,
                                                       self.backward)
        else:
            residual, stats = tf.compose_vector_fields3D(self.forward,
                                                         self.backward)
        return residual, stats
