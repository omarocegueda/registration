import numpy as np
import registrationCommon as rcommon
import ecqmmf
import matplotlib.pyplot as plt
import nibabel as nib
import ecqmmf_reg

def initializeECQMMFRegistration(fixedImage, movingImage, nclasses, lambdaParam, mu, maxIter, tolerance):
    meansFixed, variancesFixed=ecqmmf.initialize_constant_models(fixedImage, nclasses)
    meansFixed=np.array(meansFixed)
    variancesFixed=np.array(variancesFixed)
    meansMoving, variancesMoving=ecqmmf.initialize_constant_models(movingImage, nclasses)
    meansMoving=np.array(meansMoving)
    variancesMoving=np.array(variancesMoving)
    segmentedFixed, meansFixed, variancesFixed, probsFixed=ecqmmf.ecqmmf(
                            fixedImage, nclasses, lambdaParam, mu, maxIter, tolerance)
    segmentedMoving, meansMoving, variancesMoving, probsMoving=ecqmmf.ecqmmf(
                            movingImage, nclasses, lambdaParam, mu, maxIter, tolerance)
    probsFixed=np.array(probsFixed)
    probsMoving=np.array(probsMoving)
    ecqmmf.update_variances(fixedImage, probsFixed, meansFixed, variancesFixed)
    ecqmmf.update_variances(movingImage, probsMoving, meansMoving, variancesMoving)
    #show variances
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(variancesFixed)
    plt.subplot(2,1,2)
    plt.plot(variancesMoving)
    #show mean images
    fixedSmooth=probsFixed.dot(meansFixed)
    movingSmooth=probsMoving.dot(meansMoving)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(fixedSmooth, cmap=plt.cm.gray)
    plt.title('Mean fixed')
    plt.subplot(1,2,2)
    plt.imshow(movingSmooth,cmap=plt.cm.gray)
    plt.title('Mean moving')
    #show mode images
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(segmentedFixed)
    plt.title('Seg. fixed')
    plt.subplot(1,2,2)
    plt.imshow(segmentedMoving)
    plt.title('Seg. moving')
    #--------------
    joint=probsFixed[:,:,:,None]*probsMoving[:,:,None,:]
    return meansFixed, variancesFixed, meansMoving, variancesMoving, joint

def testMultimodalRigidTransformationMultiScale3D_ecqmmf(betaGT, level, nclasses, lambdaParam, mu, maxIter, tolerance):
#    movingName='data/t2/t2_icbm_normal_1mm_pn0_rf0.rawb'
#    fixedName='data/t1/t1_icbm_normal_1mm_pn0_rf0.rawb'
#    print 'Loading volumes...'
#    moving=np.fromfile(movingName, dtype=np.ubyte).reshape(ns,nr,nc)
#    moving=moving.astype(np.float64)
#    moving=(moving-moving.min())/(moving.max()-moving.min())
#    fixed=np.fromfile(fixedName, dtype=np.ubyte).reshape(ns,nr,nc)
#    fixed=fixed.astype(np.float64)
#    fixed=(fixed-fixed.min())/(fixed.max()-fixed.min())
    betaGTRads=np.array(betaGT, dtype=np.float64)
    betaGTRads[0:3]=np.copy(np.pi*betaGTRads[0:3]/180.0)
    movingName='data/t2/t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz'
    fixedName ='data/t1/t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz'
    print 'Loading data...'
    moving=nib.load(movingName)
    moving=moving.get_data().squeeze()
    moving=moving.astype(np.float64)
    fixed=nib.load(fixedName)
    fixed=fixed.get_data().squeeze()
    fixed=fixed.astype(np.float64)
    moving=(moving-moving.min())/(moving.max()-moving.min())
    fixed=(fixed-fixed.min())/(fixed.max()-fixed.min())
    print 'Applying GT transform...'
    fixed=rcommon.applyRigidTransformation3D(fixed, betaGTRads)    
    ns=moving.shape[0]
    nr=moving.shape[1]
    nc=moving.shape[2]
    #leftPyramid=[i for i in rcommon.pyramid_gaussian_3D(moving, level)]
    #rightPyramid=[i for i in rcommon.pyramid_gaussian_3D(fixed, level)]
    fixedImage=fixed[:,nr//2,:].copy()
    movingImage=moving[:,nr//2,:].copy()
    print 'Initializing registration...'
    meansFixed, variancesFixed, meansMoving, variancesMoving, joint=initializeECQMMFRegistration(fixedImage, movingImage, nclasses, lambdaParam, mu, maxIter, tolerance)
    joint=np.array(joint)
    negLogLikelihood=np.zeros_like(joint)
    ecqmmf_reg.compute_registration_neg_log_likelihood_constant_models(fixedImage, movingImage, joint, meansFixed, meansMoving, variancesFixed, variancesMoving, negLogLikelihood)
    #ecqmmf_reg.initialize_registration_maximum_likelihood_probs(negLogLikelihood, joint):
    ecqmmf_reg.initialize_registration_normalized_likelihood(negLogLikelihood, joint)
    bufferN=np.array(range(nclasses*nclasses))
    bufferD=np.array(range(nclasses*nclasses))
    for i in range(maxIter):
        print 'Iter:',iter_count,'/',maxIter
        ecqmmf_reg.iterate_marginals(negLogLikelihood, joint, lambdaParam, mu, bufferN, bufferD)
        ecqmmf_reg.integrate_registration_probabilistic_weighted_tensor_field_products(double[:,:,:] q, double[:,:] diff, double[:,:,:,:] probs, double[:] weights)
        ecqmmf_reg.compute_registration_neg_log_likelihood_constant_models(fixedImage, movingImage, joint, meansFixed, meansMoving, variancesFixed, variancesMoving, negLogLikelihood)
    #----plot joint probability maps---
    print 'Plotting joint probability maps...'
    plt.figure()
    plt.title('Joint probability maps')
    for i in range(nclasses):
        for j in range(nclasses):
            plt.subplot(nclasses,nclasses,1+i*nclasses+j)
            plt.imshow(joint[:,:,i,j], cmap=plt.cm.gray)
            plt.title("F="+str(i)+", M="+str(j))
    #----plot negLogLikelihood maps---
    print 'Plotting negLogLikelihood maps...'
    plt.figure()
    plt.title('neg-log-likelihood maps')
    for i in range(nclasses):
        for j in range(nclasses):
            plt.subplot(nclasses,nclasses,1+i*nclasses+j)
            plt.imshow(negLogLikelihood[:,:,i,j], cmap=plt.cm.gray)
            plt.title("F="+str(i)+", M="+str(j))
#    print 'Estimation started.'
#    beta=estimateMultiModalRigidTransformationMultiscale3D(leftPyramid, rightPyramid)
#    print 'Estimation finished.'
#    print 'Ground truth:', betaGT
#    plotSlicePyramidsAxial(leftPyramid, rightPyramid)
#    return beta

if __name__=='__main__':    
    deg=10.0
    betaGT=np.array([0.0, deg, 0.0, 0.0, 0.0, 0.0])
    #betaGT=np.array([4.0, -4.0, 4.0, 0.0, 0.0, 0.0])
    level=1
    nclasses=3
    lambdaParam=0.01
    mu=0.005
    maxIter=100
    tolerance=1e-5
    testMultimodalRigidTransformationMultiScale3D_ecqmmf(betaGT, level, nclasses, lambdaParam, mu, maxIter, tolerance)