# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:26:13 2013

@author: khayyam
"""
def plotTrustRegions(inImg, level, dTheta, thresholds):
    left=ndimage.rotate(inImg, dTheta)
    right=ndimage.rotate(inImg, -dTheta)
    rightPyramid=[i for i in transform.pyramid_gaussian(right, level)]
    leftPyramid=[i for i in transform.pyramid_gaussian(left, level)]
    sel=level
    left=leftPyramid[sel]
    right=rightPyramid[sel]
    center=(np.array(right.shape)-1)/2.0
    C,R=sp.meshgrid(np.array(range(right.shape[1]), dtype=np.float64), np.array(range(right.shape[0]), dtype=np.float64))
    R=R-center[0]
    C=C-center[1]
    #maxAngle=2.0*dTheta
    maxAngle=8
    thetaRads=maxAngle*np.pi/180.0
    N=np.sqrt((R*thetaRads)**2+(C*thetaRads)**2)
    nt=len(thresholds)
    plt.figure()
    i=0
    for i in range(len(thresholds)):
        M=(N<thresholds[i])
        plt.subplot(1, nt, i+1)
        plt.imshow(left*M)
        plt.title('Threshold='+str(thresholds[i]))

#def testNipyRegistration(betaGT):
#    betaGTRads=np.array(betaGT, dtype=np.float64)
#    betaGTRads[0:3]=np.copy(np.pi*betaGTRads[0:3]/180.0)
#    ns=181
#    nr=217
#    nc=181
#    print 'Loading volume...'
#    inImg=np.fromfile('data/t2/t2_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape(ns,nr,nc)
#    inImg=inImg.astype(np.float64)
#    left=inImg    
#    right=applyRigidTransformation3D(inImg, -1*betaGTRads)
#    affine_transform=AffineTransform('ijk', ['aligned-z=I->S','aligned-y=P->A', 'aligned-x=L->R'], np.eye(4))
#    left=Image(left, affine_transform)
#    right=Image(right, affine_transform)
#    print 'Estimation started.'
#    reggie=HistogramRegistration(right, left)
#    aff = reggie.optimize('rigid').as_affine()
#    print 'Estimation finished.'
#    return aff
    
#    if(np.abs(dTheta[0])>epsilon):
#        left=ndimage.rotate(inImg, -0.5*dTheta[0], axes=(1,2))
#        right=ndimage.rotate(inImg, 0.5*dTheta[0], axes=(1,2))
    #if(np.abs(dTheta[1])>epsilon):
#        left=ndimage.rotate(inImg, -0.5*dTheta[1], axes=(0,2))
#        right=ndimage.rotate(inImg, 0.5*dTheta[1], axes=(0,2))
    #if(np.abs(dTheta[2])>epsilon):
#        left=ndimage.rotate(inImg, -0.5*dTheta[2], axes=(0,1))
#        right=ndimage.rotate(inImg, 0.5*dTheta[2], axes=(0,1))    

def estimateNewSingleAngle3D(left, right, naxis, previousBeta=None):
    epsilon=1e-9
    sh=right.shape
    center=(np.array(sh)-1)/2.0
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
    X0=X0-center[0]
    X1=X1-center[1]
    X2=X2-center[2]
    if((previousBeta!=None) and (np.max(np.abs(previousBeta))>epsilon)):
        R=getRotationMatrix(previousBeta[0:3])
        X0new,X1new,X2new=(R[0,0]*X0 + R[0,1]*X1 + R[0,2]*X2 + center[0] , 
                        R[1,0]*X0 + R[1,1]*X1 + R[1,2]*X2 + center[1] , 
                        R[2,0]*X0 + R[2,1]*X1 + R[2,2]*X2 + center[2])
        right=ndimage.map_coordinates(right, [X0new, X1new, X2new])
    g0, g1, g2=sp.gradient(right)
    q=None
    if(naxis==0):
        q=g2*X1-g1*X2
    elif(naxis==1):
        q=g0*X2-g2*X0
    elif(naxis==2):
        q=g1*X0-g0*X1
    tensorProds=q**2
    A=np.sum(tensorProds)
    diff=left-right
    prod=q*diff
    b=np.sum(prod)
    beta=b/A
    return beta

def estimateRotation3DIterative(left, right, previousBeta=None):
    epsilon=1e-3
    beta=None
    if(previousBeta==None):
        beta=np.array([0.0, 0.0, 0.0])
    else:
        beta=previousBeta
    maxAbsAngle=epsilon+1
    niter=0
    while (maxAbsAngle>epsilon):
        maxAbsAngle=0.0
        for naxis in range(3):
            angle=estimateNewSingleAngle3D(left, right, naxis, beta)
            if(angle>0.17):
                angle=0.17
            elif(angle<-0.17):
                angle=-0.17
            beta[naxis]+=angle
            maxAbsAngle=np.max([maxAbsAngle, np.abs(angle)])
        niter+=1
        print niter, maxAbsAngle
    return beta

def estimateRotation3DIterativeMultiscale(leftPyramid, rightPyramid, level=0, paramList=None):
    n=len(leftPyramid)
    if(level>=n):
        return np.array([0,0,0], dtype=np.float64)
    if(level==(n-1)):
        beta=estimateNewRigidTransformation3D(leftPyramid[level], rightPyramid[level])
        if(paramList!=None):
            paramList.append(beta)
        return beta
    betaSub=estimateRotation3DIterativeMultiscale(leftPyramid, rightPyramid, level+1, paramList)
    beta=estimateRotation3DIterative(leftPyramid[level], rightPyramid[level], betaSub)
    if(paramList!=None):
        paramList.append(beta)
    beta=beta+(betaSub*np.array([1.0, 1.0, 1.0]))
    print 180*beta/np.pi
    return beta

def testEstimateRotation3DIterativeMultiscale(betaGT, level):
    betaGTRads=np.array(betaGT, dtype=np.float64)
    betaGTRads[0:3]=np.copy(np.pi*betaGTRads[0:3]/180.0)
    ns=181
    nr=217
    nc=181
    inImg=np.fromfile('data/t2/t2_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape(ns,nr,nc)
    inImg=inImg.astype(np.float64)
    #inImg=inImg[60:151,:,:]
    left=inImg
    right=applyRigidTransformation3D(inImg, -1*betaGTRads)
    leftPyramid=[i for i in pyramid_gaussian_3D(left, level)]
    rightPyramid=[i for i in pyramid_gaussian_3D(right, level)]
    left=leftPyramid[4]
    right=rightPyramid[4]
    plotSlicePyramidsAxial(leftPyramid, rightPyramid)
    estimateNewSingleAngle3D(left, right, 0)*180/np.pi
    beta=estimateRotation3DIterativeMultiscale(leftPyramid, rightPyramid)
    print 180.0*beta/np.pi
    return beta


    
def generateTestingPair():
    betaGTRads=np.array(betaGT, dtype=np.float64)
    betaGTRads[0:3]=np.copy(np.pi*betaGTRads[0:3]/180.0)
    ns=181
    nr=217
    nc=181
    inImg=np.fromfile('data/t2/t2_icbm_normal_1mm_pn0_rf0.rawb', dtype=np.ubyte).reshape(ns,nr,nc)
    inImg=inImg.astype(np.float64)
    left=inImg
    right=applyRigidTransformation3D(inImg, -1*betaGTRads)
    affine_transform=AffineTransform('ijk', ['aligned-z=I->S','aligned-y=P->A', 'aligned-x=L->R'], np.eye(4))
    left=Image(left, affine_transform)
    right=Image(right, affine_transform)
    nipy.save_image(left,'moving.nii')
    nipy.save_image(right,'fixed.nii')
