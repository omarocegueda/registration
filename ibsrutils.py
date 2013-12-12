#!/opt/python/anaconda/bin/python
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import itertools
import tensorFieldUtils as tf
import sys
import os
import registrationCommon as rcommon

def changeExtension(fname, newExt):
    '''
    changeExtension('/opt/registration/data/myfile.nii.gz', '.ext')
    changeExtension('/opt/registration/data/myfile.nii.gz', '_suffix.ext')
    '''
    directory=os.path.dirname(fname)
    if directory:
        directory+='/'
    basename=rcommon.getBaseFileName(fname)
    return directory+basename+newExt

def getSegmentationStats(namesFile):
    '''
    cnt, sizes, common=getSegmentationStats('/opt/registration/experiments/segNames.txt')
    '''
    names=None
    with open(namesFile) as f:
        names = f.readlines()
    i=0
    cnt=np.zeros(shape=(256,), dtype=np.int32);
    sizes=np.zeros(shape=(len(names), 256), dtype=np.int32)
    for name in names:
        name=name.strip()
        nib_vol = nib.load(name)
        vol=nib_vol.get_data().squeeze().astype(np.int32).reshape(-1)
        vol.sort()
        values = list(set(vol))
        groups={g[0]:len(list(g[1])) for g in itertools.groupby(vol)}
        for key, val in groups.iteritems():
            sizes[i,key]=val
        for val in values:
            cnt[val]+=1
        print i,len(values)
        i+=1
    common=np.array(range(256))[cnt==18]
    return cnt, sizes, common

def getLabelingInfo(fname):
    '''
    labels, colors=getLabelingInfo('/opt/registration/data/IBSR_labels.txt')
    '''
    with open(fname) as f:
        lines=f.readlines()
    colors={}
    labels={}
    for line in lines:
        items=line.split()
        if not items:
            break
        colors[int(items[0])]=(float(items[2])/255.0, float(items[3])/255.0, float(items[4])/255.0)
        labels[int(items[0])]=items[1]
    return labels, colors

def plotRegionsizes(namesFile, labelInfoFname):
    '''
    sizes, labels, colors=plotRegionsizes('/opt/registration/experiments/segNames.txt', '/opt/registration/data/IBSR_labels.txt')
    '''
    labels, colors=getLabelingInfo(labelInfoFname)
    cnt, sizes, common=getSegmentationStats(namesFile)
    cc=[colors[i] for i in common]
    names=[labels[i] for i in common]
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_color_cycle(cc)
    plots=[]
    for i in common:
        p,=plt.plot(sizes[:,i], color=colors[i])
        plots.append(p)
    plt.legend(plots, names)
    return sizes, labels, colors

def segmentBrainwebAtlas(segNames, displacementFnames):
    nvol=len(segNames)
    votes=None
    for i in range(nvol):
        segNames[i]=segNames[i].strip()
        displacementFnames[i]=displacementFnames[i].strip()
        nib_vol = nib.load(segNames[i])
        vol=nib_vol.get_data().squeeze().astype(np.int32)
        vol=np.copy(vol, order='C')
        displacement=np.load(displacementFnames[i])
        print 'Warping segmentation', i+1, '/',nvol, '. Vol shape:', vol.shape, 'Disp. shape:', displacement.shape
        warped=np.array(tf.warp_discrete_volumeNN(vol, displacement))
        del vol
        del displacement
        if votes==None:
            votes=np.ndarray(shape=(warped.shape+(nvol,)), dtype=np.int32)
        votes[...,i]=warped
        del warped
    print 'Computing voting segmentation...'
    seg=np.array(tf.get_voting_segmentation(votes)).astype(np.int16)
    img=nib.Nifti1Image(seg, np.eye(4))
    print 'Saving segmentation...'
    img.to_filename('votingSegmentation.nii.gz')

def warpSegmentation(segName, dispName):
    nib_vol = nib.load(segName)
    vol=nib_vol.get_data().squeeze().astype(np.int32)
    vol=np.copy(vol, order='C')
    displacement=np.load(dispName)
    baseName=rcommon.getBaseFileName(segName)
    warped=np.array(tf.warp_discrete_volumeNN(vol, displacement))
    warped=nib.Nifti1Image(warped, np.eye(4))
    warped.to_filename("warped"+baseName+"nii.gz")

def showRegistrationResultMidSlices(fnameMoving, fnameFixed, fnameAffine=None):
    '''
    showRegistrationResultMidSlices('IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 't1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')
    showRegistrationResultMidSlices('warpedDiff_IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeled_t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 't1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')
    showRegistrationResultMidSlices('IBSR_04_ana_strip_t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 't2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')
    showRegistrationResultMidSlices('warpedDiff_IBSR_04_ana_strip_t2_icbm_normal_1mm_pn0_rf0_peeled_t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 't2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')
    
    
    showRegistrationResultMidSlices('IBSR_01_ana_strip_t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 't2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')
    showRegistrationResultMidSlices('warpedDiff_IBSR_01_ana_strip_t2_icbm_normal_1mm_pn0_rf0_peeled_t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 't2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')
    
    showRegistrationResultMidSlices('t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 'IBSR_01_ana_strip_t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')        
    showRegistrationResultMidSlices('warpedDiff_t2_icbm_normal_1mm_pn0_rf0_peeled_IBSR_01_ana_strip_t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 'IBSR_01_ana_strip_t2_icbm_normal_1mm_pn0_rf0_peeled.nii.gz')    
    
    showRegistrationResultMidSlices('IBSR_01_ana_strip.nii.gz', 't1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 'IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt')
    showRegistrationResultMidSlices('warpedDiff_IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 't1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', None)
    
    
    showRegistrationResultMidSlices('warpedDiff_IBSR_01_ana_strip_IBSR_02_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz', None)
    showRegistrationResultMidSlices('warpedDiff_IBSR_01_segTRI_ana_IBSR_02_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_02/IBSR_02_segTRI_ana.nii.gz', None)
    '''
    if(fnameAffine==None):
        T=np.eye(4)
    else:
        T=rcommon.readAntsAffine(fnameAffine)
    fixed=nib.load(fnameFixed)
    F=fixed.get_affine()
    fixed=fixed.get_data().squeeze().astype(np.float64)
    moving=nib.load(fnameMoving)
    M=moving.get_affine()
    moving=moving.get_data().squeeze().astype(np.float64)
    initAffine=np.linalg.inv(M).dot(T.dot(F))
    
    fixed=np.copy(fixed, order='C')
    moving=np.copy(moving, order='C')
    warped=np.array(tf.warp_volume_affine(moving, np.array(fixed.shape), initAffine))
    sh=warped.shape
    rcommon.overlayImages(warped[:,sh[1]//2,:], fixed[:,sh[1]//2,:])
    rcommon.overlayImages(warped[sh[0]//2,:,:], fixed[sh[0]//2,:,:])
    rcommon.overlayImages(warped[:,:,sh[2]//2], fixed[:,:,sh[2]//2])    

def computeJacard(aname, bname):
    nib_A=nib.load(aname)
    affineA=nib_A.get_affine()
    A=nib_A.get_data().squeeze().astype(np.int32)
    A=np.copy(A, order='C')
    print "A range:",A.min(), A.max()
    nib_B=nib.load(bname)
    newB=nib.Nifti1Image(nib_B.get_data(),affineA)
    newB.to_filename(bname)
    B=nib_B.get_data().squeeze().astype(np.int32)
    B=np.copy(B, order='C')
    print "B range:",B.min(), B.max()
    nlabels=1+np.max([A.max(), B.max()])
    jacard=np.array(tf.compute_jacard(A,B, nlabels))
    print "Jacard range:",jacard.min(), jacard.max()
    baseA=rcommon.getBaseFileName(aname)
    baseB=rcommon.getBaseFileName(bname)
    np.savetxt("jacard_"+baseA+"_"+baseB+".txt",jacard)
    return jacard

if __name__=="__main__":
    argc=len(sys.argv)
    if argc<2:
        print 'Task name expected:\n','segatlas\n','invert\n','npy2nifti\n','lattice\n'
        sys.exit(0)
    if(sys.argv[1]=='segatlas'):
        if argc<4:
            print "Two file names expected as arguments: segmentation files and displacement files"
            sys.exit(0)
        segNamesFile=sys.argv[2]
        dispNamesFile=sys.argv[3]
        try:
            with open(segNamesFile) as f:
                segNames=f.readlines()
        except IOError:
            print 'Cannot open file:',segNamesFile
            sys.exit(0)
        try:
            with open(dispNamesFile) as f:
                displacementFnames=f.readlines()
        except IOError:
            print 'Cannot open file:',dispNamesFile
            sys.exit(0)
        segmentBrainwebAtlas(segNames, displacementFnames)
        sys.exit(0)
    elif(sys.argv[1]=='invert'):
        if argc<3:
            print 'Displacement-field file name expected.'
            sys.exit(0)
        dispName=sys.argv[2]
        displacement=np.load(dispName)
        lambdaParam=0.9
        maxIter=100
        tolerance=1e-4
        if argc>3:
            lambdaParam=float(sys.argv[3])
        if argc>4:
            maxIter=int(sys.argv[4])
        if argc>5:
            tolerance=float(sys.argv[5])
        print 'Inverting displacement: ',dispName, '. With parameters: lambda=',lambdaParam, '. Maxiter=',maxIter, '. Tolerance=',tolerance,'...'
        inverse=np.array(tf.invert_vector_field3D(displacement, lambdaParam, maxIter, tolerance))
        invName="inv"+dispName
        print 'Saving inverse as:', invName
        np.save(invName, inverse)
        print 'Computing inversion error...'
        residual=np.array(tf.compose_vector_fields3D(displacement, inverse))
        residualName="res"+dispName
        print 'Saving residual as:', residualName
        np.save(residualName, residual)
        residual=np.sqrt(np.sum(residual**2,3))
        print "Mean residual norm:", residual.mean()," (",residual.std(), "). Max residual norm:", residual.max()
        sys.exit(0)
    elif(sys.argv[1]=='npy2nifti'):
        if argc<3:
            print 'File name expected.'
            sys.exit(0)
        fname=sys.argv[2]
        try:
            inputData=np.load(fname)
        except IOError:
            print 'Cannot open file:',fname
            sys.exit(0)
        outputData=nib.Nifti1Image(inputData, np.eye(4))
        outputName=changeExtension(fname, '.nii.gz')
        outputData.to_filename(outputName)
        sys.exit(0)
    elif(sys.argv[1]=='lattice'):
        if argc<3:
            print 'File name expected.'
            sys.exit(0)
        dname=sys.argv[2]
        oname='lattice_'+changeExtension(dname, '.nii.gz')
        rcommon.saveDeformedLattice3D(dname, oname)
        sys.exit(0)
    elif(sys.argv[1]=='warpseg'):
        if argc<4:
            print "Two file names expected as arguments"
            sys.exit(0)
        segName=sys.argv[2]
        dispName=sys.argv[3]
        print "Warping:", segName
        warpSegmentation(segName, dispName)
    elif(sys.argv[1]=='jacard'):
        if argc<5:
            print "Two file names and a numerical parameter (num labels) expected as arguments"
            sys.exit(0)
        aname=sys.argv[2]
        bname=sys.argv[3]
        computeJacard(aname, bname)
        sys.exit(0)
    elif(sys.argv[1]=='fulljacard'):#compute the mean and std dev of jacard index among all pairs of the given volumes
        if argc<3:
            print "A text file containing the segmentation names must be provided."
        try:
            with open(sys.argv[2]) as f:
                names=[line.strip().split() for line in f.readlines()]
        except IOError:
            print 'Cannot open file:',sys.argv[2]
            sys.exit(0)
        nlines=len(names)
        sumJacard=None
        sumJacard2=None
        nsamples=0.0
        for i in range(nlines):
            if not names[i]:
                continue
            registrationReference=names[i][0]
            reference=names[i][1]
            for j in range(nlines):
                if i==j:
                    continue
                if not names[j]:
                    continue
                target=names[j][1]
                ###############
                baseReference=rcommon.getBaseFileName(registrationReference)
                baseTarget=rcommon.getBaseFileName(target)
                warpedName='warpedDiff_'+baseTarget+'_'+baseReference+'.nii.gz'
                jacard=computeJacard(reference, warpedName)
                nsamples+=1
                if sumJacard==None:
                    sumJacard=jacard
                    sumJacard2=jacard**2
                else:
                    shOld=sumJacard.shape
                    shNew=jacard.shape
                    extendedShape=(np.max([shOld[0], shNew[0]]), np.max([shOld[1], shNew[1]]))
                    newSum=np.zeros(shape=extendedShape, dtype=np.float64)
                    newSum2=np.zeros(shape=extendedShape, dtype=np.float64)
                    newSum[:shOld[0], :shOld[1]]=sumJacard[...]
                    newSum[:shNew[0], :shNew[1]]+=jacard[...]
                    newSum2[:shOld[0], :shOld[1]]=sumJacard2[...]
                    newSum2[:shNew[0], :shNew[1]]+=jacard[...]**2
                    sumJacard=newSum
                    sumJacard2=newSum2
        meanJacard=sumJacard/nsamples
        variance=sumJacard2/nsamples-meanJacard**2#E[X^2] - E[X]^2
        std=np.sqrt(variance)
        np.savetxt("jacard_mean.txt",meanJacard)
        np.savetxt("jacard_std.txt",std)
        sys.exit(0)
    print 'Unknown argument:',sys.argv[1]
