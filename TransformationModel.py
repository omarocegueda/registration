import numpy as np
import tensorFieldUtils as tf

class TransformationModel(object):
    '''
    This class maps points between two spaces: "reference space" and "target space"
    Forward: maps target to reference, y=affinePost*forward(affinePre*x)
    Backward: maps reference to target, x=affinePre^{-1}*backward(affinePost^{-1}*y)
    '''
    def __init__(self, forward=None, backward=None, affinePre=None, affinePost=None):
        self.dim=None
        self.setForward(forward)
        self.setBackward(backward)       
        self.setAffinePre(affinePre)
        self.setAffinePost(affinePost)

    def setAffinePre(self, affinePre):
        if affinePre!=None:
            self.dim=affinePre.shape[1]-1
            self.affinePreInv=np.linalg.inv(affinePre)
        else:
            self.affinePreInv=None
        self.affinePre=affinePre

    def setAffinePost(self, affinePost):
        if affinePost!=None:
            self.dim=affinePost.shape[1]-1
            self.affinePostInv=np.linalg.inv(affinePost)
        else:
            self.affinePostInv=None
        self.affinePost=affinePost

    def setForward(self, forward):
        if forward!=None:
            self.dim=len(forward.shape)-1
        self.forward=forward

    def setBackward(self, backward):
        if backward!=None:
            self.dim=len(backward.shape)-1
        self.backward=backward

    def getAffinePre(self):
        return self.affinePre

    def getAffinePost(self):
        return self.affinePost

    def getForward(self):
        return self.forward

    def getBackward(self):
        return self.backward

    def warpForward(self, image):
        if len(image.shape)==3:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_volume(image, self.forward, self.affinePre, self.affinePost))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_volume(image, self.forward, self.affinePre, self.affinePost))
        else:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_image(image, self.forward, self.affinePre, self.affinePost))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_image(image, self.forward, self.affinePre, self.affinePost))
        return warped

    def warpBackward(self, image):
        if len(image.shape)==3:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_volume(image, self.backward, self.affinePostInv, self.affinePreInv))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_volume(image, self.backward, self.affinePostInv, self.affinePreInv))
        else:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_image(image, self.backward, self.affinePostInv, self.affinePreInv))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_image(image, self.backward, self.affinePostInv, self.affinePreInv))
        return warped

    def warpForwardNN(self, image):
        if len(image.shape)==3:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_volumeNN(image, self.forward, self.affinePre, self.affinePost))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_volumeNN(image, self.forward, self.affinePre, self.affinePost))
        else:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_imageNN(image, self.forward, self.affinePre, self.affinePost))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_imageNN(image, self.forward, self.affinePre, self.affinePost))
        return warped

    def warpBackwardNN(self, image):
        if len(image.shape)==3:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_volumeNN(image, self.backward, self.affinePostInv, self.affinePreInv))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_volumeNN(image, self.backward, self.affinePostInv, self.affinePreInv))
        else:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_imageNN(image, self.backward, self.affinePostInv, self.affinePreInv))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_imageNN(image, self.backward, self.affinePostInv, self.affinePreInv))
        return warped

    def __scaleAffine(self, affine, factor):
        scaledAffine=affine.copy()
        n=affine.shape[1]-1
        scaledAffine[:n,n]*=factor
        return scaledAffine

    def scaleAffines(self, factor):
        if self.affinePre!=None:
            self.affinePre=self.__scaleAffine(self.affinePre, factor)
            self.affinePreInv=np.linalg.inv(self.affinePre)
        if self.affinePost!=None:
            self.affinePost=self.__scaleAffine(self.affinePost, factor)
            self.affinePostInv=np.linalg.inv(self.affinePost)

    def upsample(self, newDomainForward, newDomainBackward):
        if self.dim==2:
            if self.forward!=None:
                self.forward=np.array(tf.upsample_displacement_field(self.forward, np.array(newDomainForward).astype(np.int32)))*2
            if self.backward!=None:
                self.backward=np.array(tf.upsample_displacement_field(self.backward, np.array(newDomainBackward).astype(np.int32)))*2
        else:
            if self.forward!=None:
                self.forward=np.array(tf.upsample_displacement_field3D(self.forward, np.array(newDomainForward).astype(np.int32)))*2
            if self.backward!=None:
                self.backward=np.array(tf.upsample_displacement_field3D(self.backward, np.array(newDomainBackward).astype(np.int32)))*2
        self.scaleAffines(2.0)
    
    def computeInversionError(self):
        if self.dim==2:
            residual, stats=tf.compose_vector_fields(self.forward, self.backward)
        else:
            residual, stats=tf.compose_vector_fields3D(self.forward, self.backward)
        return residual, stats
