# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:55:17 2013

@author: khayyam
"""
import nibabel as nib
import matplotlib.pyplot as plt
import ecqmmf
import numpy as np
image=nib.load('data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz')
image=image.get_data().squeeze()
image=image.astype(np.float64)

sh=image.shape
#img=image[:,:,sh[2]//2].copy()
img=image[:,sh[1]//2,:].copy()
plt.imshow(img, cmap=plt.cm.gray)

nclasses=25
#lambdaParam=150.0
lambdaParam=25.0
mu=5.0
maxIter=100
tolerance=1e-5

means, variances=ecqmmf.initialize_constant_models(img, nclasses)
means=np.array(means)
variances=np.array(variances)

segmented, means, variances, probs=ecqmmf.ecqmmf(img, nclasses, lambdaParam, mu, maxIter, tolerance)
segmented=np.array(segmented)
means=np.array(means)
variances=np.array(variances)
probs=np.array(probs)
plt.figure()
plt.imshow(segmented)
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

