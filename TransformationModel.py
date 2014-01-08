import numpy as np
import tensorFieldUtils as tf

class TransformationModel(object):
    def __init__(self, forward=None, backward=None, affineFixed=None, affineMoving=None):
        self.dim=None
        self.setForward(forward)
        self.setBackward(backward)       
        self.setAffineFixed(affineFixed)
        self.setAffineMoving(affineMoving)

    def setAffineFixed(self, affineFixed):
        if affineFixed!=None:
            self.dim=affineFixed.shape[1]-1
        self.affineFixed=affineFixed

    def setAffineMoving(self, affineMoving):
        if affineMoving!=None:
            self.dim=affineMoving.shape[1]-1
        self.affineMoving=affineMoving

    def setForward(self, forward):
        if forward!=None:
            self.dim=len(forward.shape)-1
        self.forward=forward

    def setBackward(self, backward):
        if backward!=None:
            self.dim=len(backward.shape)-1
        self.backward=backward

    def getAffineFixed(self):
        return self.affineFixed

    def getAffineMoving(self):
        return self.affineMoving

    def getForward(self):
        return self.forward

    def getBackward(self):
        return self.backward

    def warpForward(self, image):
        if len(image.shape)==3:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_volume(image, self.forward, self.affineMoving))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_volume(image, self.forward, self.affineMoving))
        else:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_image(image, self.forward, self.affineMoving))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_image(image, self.forward, self.affineMoving))
        return warped

    def warpBackward(self, image):
        if len(image.shape)==3:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_volume(image, self.backward, self.affineFixed))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_volume(image, self.backward, self.affineFixed))
        else:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_image(image, self.backward, self.affineFixed))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_image(image, self.backward, self.affineFixed))
        return warped

    def warpForwardNN(self, image):
        if len(image.shape)==3:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_volumeNN(image, self.forward, self.affineMoving))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_volumeNN(image, self.forward, self.affineMoving))
        else:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_imageNN(image, self.forward, self.affineMoving))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_imageNN(image, self.forward, self.affineMoving))
        return warped

    def warpBackwardNN(self, image):
        if len(image.shape)==3:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_volumeNN(image, self.backward, self.affineFixed))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_volumeNN(image, self.backward, self.affineFixed))
        else:
            if image.dtype is np.dtype('int32'):
                warped=np.array(tf.warp_discrete_imageNN(image, self.backward, self.affineFixed))
            elif image.dtype is np.dtype('float64'):
                warped=np.array(tf.warp_imageNN(image, self.backward, self.affineFixed))
        return warped

    def __scaleAffine(self, affine, factor):
        scaledAffine=affine.copy()
        n=affine.shape[1]-1
        scaledAffine[:n,n]*=factor
        return scaledAffine

    def scaleAffines(self, factor):
        if self.affineMoving!=None:
            self.affineMoving=self.__scaleAffine(self.affineMoving, factor)
        if self.affineFixed!=None:
            self.affineFixed=self.__scaleAffine(self.affineFixed, factor)
        
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
