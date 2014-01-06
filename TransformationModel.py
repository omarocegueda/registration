import numpy as np
import tensorFieldUtils as tf

class TransformationModel(object):
    def __init__(self, forward=None, backward=None, affineFixed=None, affineMoving=None):
        print forward, backward, affineFixed, affineMoving
        self.forward=forward
        self.backward=backward
        self.affineFixed=affineFixed
        self.affineMoving=affineMoving

    def setAffineFixed(self, affineFixed):
        self.affineFixed=affineFixed

    def setAffineMoving(self, affineMoving):
        self.affineMoving=affineMoving

    def setForward(self, forward):
        self.forward=forward

    def setBackward(self, backward):
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
            warped=np.array(tf.warp_volume(image, self.forward, self.affineMoving))
        else:
            warped=np.array(tf.warp_image(image, self.forward, self.affineMoving))
        return warped

    def warpBackward(self, image):
        if len(image.shape)==3:
            warped=np.array(tf.warp_volume(image, self.backward, self.affineFixed))
        else:
            warped=np.array(tf.warp_image(image, self.backward, self.affineFixed))
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
        if self.forward!=None:
            self.forward=np.array(tf.upsample_displacement_field(self.forward, np.array(newDomainForward)))*2
        if self.backward!=None:
            self.backward=np.array(tf.upsample_displacement_field(self.backward, np.array(newDomainBackward)))*2
        self.scaleAffines(2.0)
