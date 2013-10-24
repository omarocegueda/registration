import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import nibabel as nib
import registrationCommon as rcommon
import ecqmmf
import ecqmmf_reg
from scipy import ndimage
from registrationCommon import const_prefilter_map_coordinates

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
    sh=moving.shape
    #leftPyramid=[i for i in rcommon.pyramid_gaussian_3D(moving, level)]
    #rightPyramid=[i for i in rcommon.pyramid_gaussian_3D(fixed, level)]
    fixedImage=fixed[:,sh[1]//2,:].copy()
    movingImage=moving[:,sh[1]//2,:].copy()
    print 'Initializing registration...'
    meansFixed, variancesFixed, meansMoving, variancesMoving, joint=initializeECQMMFRegistration(fixedImage, movingImage, nclasses, lambdaParam, mu, maxIter, tolerance)
    joint=np.array(joint)
    negLogLikelihood=np.zeros_like(joint)
    ecqmmf_reg.compute_registration_neg_log_likelihood_constant_models(fixedImage, movingImage, joint, meansFixed, meansMoving, variancesFixed, variancesMoving, negLogLikelihood)
    #ecqmmf_reg.initialize_registration_maximum_likelihood_probs(negLogLikelihood, joint):
    ecqmmf.initialize_normalized_likelihood(negLogLikelihood, joint)
    bufferN=np.array(range(nclasses*nclasses))
    bufferD=np.array(range(nclasses*nclasses))
    for iter_count in range(maxIter):
        print 'Iter:',iter_count,'/',maxIter
        ecqmmf_reg.iterate_marginals(negLogLikelihood, joint, lambdaParam, mu, bufferN, bufferD)
        #ecqmmf_reg.integrate_registration_probabilistic_weighted_tensor_field_products(double[:,:,:] q, double[:,:] diff, double[:,:,:,:] probs, double[:] weights)
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

###############################################################
####### Non-linear Multimodal registration - EM (2D)###########
###############################################################

def estimateNewECQMMFMultimodalDeformationField2D(fixed, moving, nclasses, lambdaMeasureField, lambdaDisplacement, mu, maxOuterIter, maxInnerIter, tolerance, previousDisplacement=None):
    sh=fixed.shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]
    displacement     =np.empty(shape=(fixed.shape)+(2,), dtype=np.float64)
    gradientField    =np.empty(shape=(fixed.shape)+(2,), dtype=np.float64)
    totalDisplacement=np.zeros(shape=(fixed.shape)+(2,), dtype=np.float64)
    gradientField    =np.empty(shape=(fixed.shape)+(2,), dtype=np.float64)
    residuals        =np.zeros_like(fixed)
    warped=None
    if(previousDisplacement!=None):
        totalDisplacement[...]=previousDisplacement
        warped=ndimage.map_coordinates(moving, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1]], prefilter=True)
    else:
        warped=moving
    #run soft segmentation on the fixed image
    meansFixed, variancesFixed=ecqmmf.initialize_constant_models(fixed, nclasses)
    meansFixed=np.array(meansFixed)
    variancesFixed=np.array(variancesFixed)
    segFixed, meansFixed, variancesFixed, probsFixed=ecqmmf.ecqmmf(fixed, nclasses, lambdaMeasureField, mu, maxOuterIter, maxInnerIter, tolerance)
    meansFixed=np.array(meansFixed)
    probsFixed=np.array(probsFixed)
    #run soft segmentation on the warped image
    meansWarped, variancesWarped=ecqmmf.initialize_constant_models(warped, nclasses)
    meansWarped=np.array(meansWarped)
    variancesWarped=np.array(variancesWarped)
    segWarped, meansWarped, variancesWarped, probsWarped=ecqmmf.ecqmmf(warped, nclasses, lambdaMeasureField, mu, maxOuterIter, maxInnerIter, tolerance)
    meansWarped=np.array(meansWarped)
    probsWarped=np.array(probsWarped)
    #inicialize the joint models (solve assignment problem)
    ecqmmf_reg.initialize_coupled_constant_models(probsFixed, probsWarped, meansWarped)
    #start optimization
    outerIter=0
    negLogLikelihood=np.zeros_like(probsFixed)
    while(outerIter<maxOuterIter):
        outerIter+=1
        print "Outer:", outerIter
        if(outerIter>1):#avoid warping twice at the first iteration
            warped=ndimage.map_coordinates(moving, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1]], prefilter=True)
        movingMask=(moving>0)*1.0
        warpedMask=ndimage.map_coordinates(movingMask, [X0+totalDisplacement[...,0], X1+totalDisplacement[...,1]], order=0, prefilter=False)
        warpedMask=warpedMask.astype(np.int32)
        #--- optimize the measure field and the intensity models ---
        ecqmmf_reg.compute_registration_neg_log_likelihood_constant_models(fixed, warped, meansFixed, meansWarped, negLogLikelihood)
        #ecqmmf.initialize_normalized_likelihood(negLogLikelihood, probsWarped)
        ecqmmf.initialize_maximum_likelihood(negLogLikelihood, probsWarped);
        innerIter=0
        mse=0
        while(innerIter<maxInnerIter):
            innerIter+=1
            print "\tInner:",innerIter
            ecqmmf.optimize_marginals(negLogLikelihood, probsWarped, lambdaMeasureField, mu, maxInnerIter, tolerance)
            mseFixed=ecqmmf.update_constant_models(fixed, probsWarped, meansFixed, variancesFixed)
            mseWarped=ecqmmf.update_constant_models(warped, probsWarped, meansWarped, variancesWarped)
            mse=np.max([mseFixed, mseWarped])
            if(mse<tolerance):
                break
        #---given the intensity models and the measure field, compute the displacement
        deltaField=meansWarped[None, None, :]-warped[:,:,None]
        gradientField[:,:,0], gradientField[:,:,1]=sp.gradient(warped)
        maxDisplacement=ecqmmf_reg.optimize_ECQMMF_displacement_field_2D(deltaField, gradientField, probsWarped, lambdaDisplacement, displacement, residuals, maxInnerIter, tolerance)
        totalDisplacement+=displacement
        if(maxDisplacement<tolerance):
            break
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(fixed, cmap=plt.cm.gray)
    plt.title("fixed")
    plt.subplot(2,2,2)
    plt.imshow(warped, cmap=plt.cm.gray)
    plt.title("moving")
    plt.subplot(2,2,3)
    plt.imshow(probsFixed.dot(meansFixed), cmap=plt.cm.gray)
    plt.title("E[fixed]")
    plt.subplot(2,2,4)
    plt.imshow(probsWarped.dot(meansWarped), cmap=plt.cm.gray)
    plt.title("E[moving]")
    return totalDisplacement

def estimateECQMMFMultimodalDeformationField2DMultiScale(fixedPyramid, movingPyramid, lambdaMeasureField, lambdaDisplacement, mu, maxOuterIter, maxInnerIter, tolerance, level=0, displacementList=None):
    n=len(fixedPyramid)
    nclasses=np.max([4,32//(2**level)])
    if(level==(n-1)):
        displacement=estimateNewECQMMFMultimodalDeformationField2D(fixedPyramid[level], movingPyramid[level], nclasses, lambdaMeasureField, lambdaDisplacement, mu, maxOuterIter, maxInnerIter, tolerance, None)
        if(displacementList!=None):
            displacementList.insert(0,displacement)
        return displacement
    subDisplacement=estimateECQMMFMultimodalDeformationField2DMultiScale(fixedPyramid, movingPyramid, lambdaMeasureField, lambdaDisplacement, mu, maxOuterIter, maxInnerIter, tolerance, level+1, displacementList)
    sh=fixedPyramid[level].shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]*0.5
    upsampled=np.empty(shape=(fixedPyramid[level].shape)+(2,), dtype=np.float64)
    upsampled[:,:,0]=ndimage.map_coordinates(subDisplacement[:,:,0], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    upsampled[:,:,1]=ndimage.map_coordinates(subDisplacement[:,:,1], [X0, X1], prefilter=const_prefilter_map_coordinates)*2
    newDisplacement=estimateNewECQMMFMultimodalDeformationField2D(fixedPyramid[level], movingPyramid[level], nclasses, lambdaMeasureField, lambdaDisplacement, mu, maxOuterIter, maxInnerIter, tolerance, upsampled)
    newDisplacement+=upsampled
    if(displacementList!=None):
        displacementList.insert(0, newDisplacement)
    return newDisplacement

def testEstimateECQMMFMultimodalDeformationField2DMultiScale_synthetic():
    ##################parameters############
    maxGTDisplacement=2
    maxPyramidLevel=0
    lambdaMeasureField=0.02
    lambdaDisplacement=200
    mu=0.001
    maxOuterIter=20
    maxInnerIter=50
    tolerance=1e-5
    displacementList=[]
    #######################################3
    #fname0='IBSR_01_to_02.nii.gz'
    #fname1='data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz'
    fnameMoving='data/t2/t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz'
    fnameFixed='data/t1/t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz'
    nib_moving = nib.load(fnameMoving)
    nib_fixed = nib.load(fnameFixed)
    moving=nib_moving.get_data().squeeze().astype(np.float64)
    fixed=nib_fixed.get_data().squeeze().astype(np.float64)
    sm=moving.shape
    sf=fixed.shape
    #---coronal---
    moving=moving[:,sm[1]//2,:].copy()
    fixed=fixed[:,sf[1]//2,:].copy()
    moving=(moving-moving.min())/(moving.max()-moving.min())
    fixed=(fixed-fixed.min())/(fixed.max()-fixed.min())
    #----apply synthetic deformation field to fixed image
    GT=rcommon.createDeformationField_type2(fixed.shape[0], fixed.shape[1], maxGTDisplacement)
    fixed=rcommon.warpImage(fixed,GT)
    maskMoving=moving>0
    maskFixed=fixed>0
    movingPyramid=[img for img in rcommon.pyramid_gaussian_2D(moving, maxPyramidLevel, maskMoving)]
    fixedPyramid=[img for img in rcommon.pyramid_gaussian_2D(fixed, maxPyramidLevel, maskFixed)]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(moving, cmap=plt.cm.gray)
    plt.title('Moving')
    plt.subplot(1,2,2)
    plt.imshow(fixed, cmap=plt.cm.gray)
    plt.title('Fixed')
    rcommon.plotOverlaidPyramids(movingPyramid, fixedPyramid)
    displacement=estimateECQMMFMultimodalDeformationField2DMultiScale(fixedPyramid, movingPyramid, lambdaMeasureField, lambdaDisplacement, mu, maxOuterIter, maxInnerIter, tolerance, 0,displacementList)
    warpedPyramid=[rcommon.warpImage(movingPyramid[i], displacementList[i]) for i in range(maxPyramidLevel+1)]
    rcommon.plotOverlaidPyramids(warpedPyramid, fixedPyramid)
    rcommon.overlayImages(warpedPyramid[0], fixedPyramid[0])
    rcommon.plotDeformationField(displacement)
    displacement[...,0]*=(maskMoving + maskFixed)
    displacement[...,1]*=(maskMoving + maskFixed)
    nrm=np.sqrt(displacement[...,0]**2 + displacement[...,1]**2)
    maxNorm=np.max(nrm)
    rcommon.plotDeformationField(displacement)
    residual=((displacement-GT))**2
    meanDisplacementError=np.sqrt(residual.sum(2)*(maskMoving + maskFixed)).mean()
    stdevDisplacementError=np.sqrt(residual.sum(2)*(maskMoving + maskFixed)).std()
    print 'Max global displacement: ', maxNorm
    print 'Mean displacement error: ', meanDisplacementError,'(',stdevDisplacementError,')'

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