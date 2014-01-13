#!/opt/python/anaconda/bin/python
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import itertools
import tensorFieldUtils as tf
import sys
import os
import registrationCommon as rcommon
from scipy import stats
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
    labels, colors=getLabelingInfo('/opt/registration/data/IBSR_common_labels.txt')
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

def warpNonlinear(targetName, referenceName, dispName, oname, interpolationType='trilinear'):
    baseName=rcommon.getBaseFileName(targetName)
    displacement=np.load(dispName)
    nib_target = nib.load(targetName)
    if interpolationType=='NN':
        target=nib_target.get_data().squeeze().astype(np.int32)
        target=np.copy(target, order='C')
        warped=np.array(tf.warp_discrete_volumeNN(target, displacement))
    else:
        target=nib_target.get_data().squeeze().astype(np.float64)
        target=np.copy(target, order='C')
        warped=np.array(tf.warp_volume(target, displacement))        
    referenceAffine=nib.load(referenceName).get_affine()
    warped=nib.Nifti1Image(warped, referenceAffine)
    if not oname:
        oname="warped"+baseName+"nii.gz"
    warped.to_filename(oname)

def warpANTSAffine(targetName, referenceName, affineName, oname, interpolationType='trilinear'):
    baseName=rcommon.getBaseFileName(targetName)
    nib_target=nib.load(targetName)
    nib_reference=nib.load(referenceName)
    M=nib_target.get_affine()
    F=nib_reference.get_affine()
    referenceShape=np.array(nib_reference.shape, dtype=np.int32)
    ######Load and compose affine#####
    if not affineName:
        T=np.eye(4)
    else:
        T=rcommon.readAntsAffine(affineName)
    affineComposition=np.linalg.inv(M).dot(T.dot(F))
    ######################
    if interpolationType=='NN':
        target=nib_target.get_data().squeeze().astype(np.int32)
        target=np.copy(target, order='C')
        warped=np.array(tf.warp_discrete_volumeNNAffine(target, referenceShape, affineComposition)).astype(np.int16)
    else:
        target=nib_target.get_data().squeeze().astype(np.float64)
        target=np.copy(target, order='C')
        warped=np.array(tf.warp_volume_affine(target, referenceShape, affineComposition)).astype(np.int16)
    warped=nib.Nifti1Image(warped, F)
    if not oname:
        oname="warped"+baseName+"nii.gz"
    warped.to_filename(oname)

def showRegistrationResultMidSlices(fnameMoving, fnameFixed, fnameAffine=None):
    '''    
    showRegistrationResultMidSlices('IBSR_01_ana_strip.nii.gz', 't1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 'IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt')
    showRegistrationResultMidSlices('warpedDiff_IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', 't1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz', None)
    
    
    showRegistrationResultMidSlices('warpedDiff_IBSR_01_ana_strip_IBSR_02_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz', None)
    showRegistrationResultMidSlices('warpedDiff_IBSR_01_segTRI_ana_IBSR_02_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_02/IBSR_02_segTRI_ana.nii.gz', None)
    ##Worst pair:
        showRegistrationResultMidSlices('warpedDiff_IBSR_16_segTRI_ana_IBSR_12_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_12/IBSR_12_segTRI_ana.nii.gz', None)
        
        showRegistrationResultMidSlices('/opt/registration/data/t1/IBSR18/IBSR_16/IBSR_16_segTRI_ana.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_12/IBSR_12_segTRI_ana.nii.gz', None)
        showRegistrationResultMidSlices('/opt/registration/data/t1/IBSR18/IBSR_16/IBSR_16_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_12/IBSR_12_ana_strip.nii.gz', None)
        showRegistrationResultMidSlices('warpedAffine_IBSR_16_segTRI_ana_IBSR_12_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_12/IBSR_12_segTRI_ana.nii.gz', None)
        showRegistrationResultMidSlices('warpedDiff_IBSR_16_ana_strip_IBSR_12_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_12/IBSR_12_ana_strip.nii.gz', None)
        showRegistrationResultMidSlices('warpedAffine_IBSR_16_ana_strip_IBSR_12_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_12/IBSR_12_ana_strip.nii.gz', None)
        
        showRegistrationResultMidSlices('/opt/registration/data/t1/IBSR18/IBSR_10/IBSR_10_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_16/IBSR_16_ana_strip.nii.gz', None)
        showRegistrationResultMidSlices('warpedAffine_IBSR_10_ana_strip_IBSR_16_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_16/IBSR_16_ana_strip.nii.gz', None)
        
        showRegistrationResultMidSlices('warpedAffine_IBSR_16_ana_strip_IBSR_10_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_10/IBSR_10_ana_strip.nii.gz', None)
        showRegistrationResultMidSlices('warpedDiff_IBSR_16_ana_strip_IBSR_10_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_10/IBSR_10_ana_strip.nii.gz', None)
        showRegistrationResultMidSlices('warpedDiff_IBSR_01_ana_strip_IBSR_08_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_08/IBSR_08_ana_strip.nii.gz', None)
        showRegistrationResultMidSlices('/opt/registration/data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_08/IBSR_08_ana_strip.nii.gz', None)
        showRegistrationResultMidSlices('warpedDiff_IBSR_13_ana_strip_IBSR_10_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_10/IBSR_10_ana_strip.nii.gz', None)
        showRegistrationResultMidSlices('/opt/registration/data/t1/IBSR18/IBSR_13/IBSR_13_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_10/IBSR_10_ana_strip.nii.gz', None)
        
        showRegistrationResultMidSlices('warpedDiff_IBSR_01_ana_strip_IBSR_02_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_10/IBSR_10_ana_strip.nii.gz', None)
        
        
        
        
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
    '''
    computeJacard('warpedDiff_IBSR_15_segTRI_fill_ana_IBSR_10_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_10/IBSR_10_segTRI_fill_ana.nii.gz' )
    computeJacard('warpedDiff_IBSR_13_segTRI_fill_ana_IBSR_10_ana_strip.nii.gz', '/opt/registration/data/t1/IBSR18/IBSR_10/IBSR_10_segTRI_fill_ana.nii.gz' )
    
    '''
    baseA=rcommon.getBaseFileName(aname)
    baseB=rcommon.getBaseFileName(bname)
    oname="jacard_"+baseA+"_"+baseB+".txt"
    if(os.path.exists(oname)):
        print 'Jacard overlap found. Skipped computation.'
        jacard=np.loadtxt(oname)
        return jacard
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
    np.savetxt(oname,jacard)
    return jacard

def fullJacard(names, segIndex, warpedPreffix):
    nlines=len(names)
    sumJacard=None
    sumJacard2=None
    minScore=None
    worstPair=None
    nsamples=0.0
    for i in range(nlines):
        if not names[i]:
            continue
        registrationReference=names[i][0]
        reference=names[i][segIndex]
        for j in range(nlines):
            if i==j:
                continue
            if not names[j]:
                continue
            target=names[j][segIndex]
            ###############
            baseReference=rcommon.getBaseFileName(registrationReference)
            baseTarget=rcommon.getBaseFileName(target)
            warpedName=warpedPreffix+baseTarget+'_'+baseReference+'.nii.gz'
            jacard=computeJacard(reference, warpedName)
            nsamples+=1
            if sumJacard==None:
                sumJacard=jacard
                sumJacard2=jacard**2
                worstPair=(i,j)
                minScore=np.sum(jacard)
            else:
                lenOld=len(sumJacard)
                lenNew=len(jacard)
                extendedShape=(np.max([lenOld, lenNew]),)
                newSum=np.zeros(shape=extendedShape, dtype=np.float64)
                newSum2=np.zeros(shape=extendedShape, dtype=np.float64)
                newSum[:lenOld]=sumJacard[...]
                newSum[:lenNew]+=jacard[...]
                newSum2[:lenOld]=sumJacard2[...]
                newSum2[:lenNew]+=jacard[...]**2
                sumJacard=newSum
                sumJacard2=newSum2
                optSum=np.sum(jacard)
                if optSum<minScore:
                    minScore=optSum
                    worstPair=(i,j)
    meanJacard=sumJacard/nsamples
    variance=sumJacard2/nsamples-meanJacard**2#E[X^2] - E[X]^2
    std=np.sqrt(variance)
    return meanJacard, std, worstPair, minScore

def getRohlfingResults(meanName, sdName):
    '''
    R=getRohlfingResults('jacard_mean_warpedDiff_3.txt', 'jacard_std_warpedDiff_3.txt')
    '''
    labels, colors=getLabelingInfo('/opt/registration/data/IBSR_common_labels.txt')
    r=np.loadtxt('data/rohlfing_table.txt')    
    means=np.loadtxt(meanName)
    sd=np.loadtxt(sdName)
    rohlfing={int(r[i,0]): np.append(r[i,1:], [[means[int(r[i,0])], sd[int(r[i,0])]]]) for i in range(r.shape[0])}
    with open('results.txt','w') as f:
        for k in rohlfing:
            line=labels[k]+':\t'+str(rohlfing[k]).replace('\n','\t').replace('[','').replace(']','')
            f.write(line+'\n')
    return rohlfing

def pairedTTests(fnames, dirA, dirB):
    labels, colors=getLabelingInfo('/home/omar/code/registration/data/IBSR_common_labels.txt')
    n=len(fnames)
    m=len(labels)
    baseline=np.ndarray(shape=(m,n), dtype=np.float64)
    follow_up=np.ndarray(shape=(m,n), dtype=np.float64)
    for i in range(n):
        fa=np.loadtxt(dirA+'/'+fnames[i])
        fb=np.loadtxt(dirB+'/'+fnames[i])
        baseline[:,i]=fa[labels.keys()]
        follow_up[:,i]=fb[labels.keys()]
    print n,',',m
    pvalues=np.ndarray((m,), dtype=np.float64)
    for i in range(m):
        t,p=stats.ttest_rel(baseline[i,:], follow_up[i,:])
        pvalues[i]=p
    return pvalues

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
    elif(sys.argv[1]=='warp' or sys.argv[1]=='warpNN'):
        if argc<5:
            print "Expected arguments: target reference dispName [oname]"
            sys.exit(0)
        targetName=sys.argv[2]
        referenceName=sys.argv[3]
        dispName=sys.argv[4]
        oname=None
        if argc>5:
            oname=sys.argv[5]
        if(sys.argv[1]=='warp'):
            warpNonlinear(targetName, referenceName, dispName, oname, 'trilinear')
        else:
            warpNonlinear(targetName, referenceName, dispName, oname, 'NN')
        sys.exit(0)
    elif(sys.argv[1]=='warpAffine' or sys.argv[1]=='warpAffineNN'):
        '''
        python ibsrutils.py warpAffineNN /opt/registration/data/t1/IBSR18/IBSR_16/IBSR_16_segTRI_ana.nii.gz /opt/registration/data/t1/IBSR18/IBSR_12/IBSR_12_ana_strip.nii.gz IBSR_16_ana_strip_IBSR_12_ana_stripAffine.txt warpedAffine_IBSR_16_segTRI_ana_IBSR_12_ana_strip.nii.gz
        '''
        if argc<5:
            print "Expected arguments: target reference affineName [oname]"
            sys.exit(0)
        targetName=sys.argv[2]
        referenceName=sys.argv[3]
        affineName=sys.argv[4]
        oname=None
        if argc>5:
            oname=sys.argv[5]
        if sys.argv[1]=='warpAffine':
            warpANTSAffine(targetName, referenceName, affineName, oname, interpolationType='trilinear')
        else:
            warpANTSAffine(targetName, referenceName, affineName, oname, interpolationType='NN')
        sys.exit(0)
    elif(sys.argv[1]=='jacard'):
        if argc<4:
            print "Two file names expected as arguments"
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
        warpedPreffix="warpedDiff_"
        if(argc>3):
            warpedPreffix=sys.argv[3]#e.g.: 'warpedAffine_'
        filesPerSample=len(names[0])
        for segIndex in range(1,filesPerSample):
            meanJacard, stdJacard, worstPair, minScore=fullJacard(names, segIndex, warpedPreffix)
            print '[', segIndex,'] Min trace:',minScore,'. Worst pair:',worstPair,'[',names[worstPair[0]][segIndex],', ',names[worstPair[1]][segIndex],']'
            np.savetxt("jacard_mean_"+warpedPreffix+str(segIndex)+'.txt',meanJacard)
            np.savetxt("jacard_std_"+warpedPreffix+str(segIndex)+'.txt',stdJacard)
        sys.exit(0)
    elif(sys.argv[1]=='ptt'):
        if argc<3:
            print "Three arguments expected: names, dirA, dirB"
        with open(sys.argv[2]) as f:
            fnames=[line.strip() for line in f.readlines()]
        dirA=sys.argv[3]
        dirB=sys.argv[4]
        pvalues=pairedTTests(fnames, dirA, dirB)
        np.savetxt('pvalues.txt',pvalues)
        
    print 'Unknown argument:',sys.argv[1]
