# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:55:17 2013

@author: khayyam
"""
import nibabel as nib
import matplotlib.pyplot as plt
import ecqmmf
import numpy as np

#imageName='data/t1/t1_icbm_normal_1mm_pn0_rf0.rawb'
#ns=181
#nr=217
#nc=181
#image=np.fromfile(imageName, dtype=np.ubyte).reshape(ns,nr,nc)
#image=image.astype(np.float64)
#img=image[90,:,:].copy()
#MM=img.max()
#mm=img.min()
#img=(img-mm)/(MM-mm)

#image=nib.load('data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz')
image=nib.load('data/t1/t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')
image=image.get_data().squeeze()
image=image.astype(np.float64)
image=(image-image.min())/(image.max()-image.min())
sh=image.shape
img=image[:,sh[1]//2,:].copy()

nclasses=16
lambdaParam=.05
mu=.01
outerIter=20
innerIter=50
tolerance=1e-6

means, variances=ecqmmf.initialize_constant_models(img, nclasses)
means=np.array(means)
variances=np.array(variances)
segmented, means, variances, probs=ecqmmf.ecqmmf(img, nclasses, lambdaParam, mu, outerIter, innerIter, tolerance)
segmented=np.array(segmented)
means=np.array(means)
variances=np.array(variances)
probs=np.array(probs)
meanEstimator=probs.dot(means)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(img)
plt.title('Input')
plt.subplot(1,3,2)
plt.imshow(segmented)
plt.title('Segmented')
plt.subplot(1,3,3)
plt.imshow(meanEstimator)
plt.title('Mean')
plt.figure()
side=np.floor((nclasses-1)**(0.5)).astype(int)+1
for i in range(nclasses):
    plt.subplot(side,side,i+1)
    plt.imshow(probs[:,:,i], cmap=plt.cm.gray)

plt.figure()
plt.subplot(1,2,1)
plt.plot(means)
plt.subplot(1,2,2)
plt.plot(variances)
print means
print variances

