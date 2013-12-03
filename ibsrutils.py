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
        vol=nib_vol.get_data().astype(np.int32).reshape(-1)
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
        nib_vol = nib.load(segNames[i])
        vol=nib_vol.get_data().astype(np.int32)
        displacement=np.load(displacementFnames[i])
        warped=np.array(tf.warp_discrete_volumeNN(vol, displacement))
        if votes==None:
            votes=np.ndarray(shape=(warped.shape+(nvol,)), dtype=np.int32)
        votes[...,i]=warped
    seg=np.array(tf.get_voting_segmentation(votes)).astype(np.int16)
    img=nib.Nifti1Image(seg, np.eye(4))
    img.to_filename('votingSegmentation.nii.gz')

def showRegistrationResultMidSlices(fnameWarped, fnameFixed):
    '''
    saveRegistrationResultMidSlices('data/affineRegistered/templateT1ToIBSR01T1.nii.gz', 'data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz', 'displacement_templateT1ToIBSR01T1_diffMulti.npy')
    '''
    fixed=nib.load(fnameFixed).get_data().squeeze().astype(np.float64)
    warped=nib.load(fnameWarped).get_data().squeeze().astype(np.float64)
    sh=warped.shape
    rcommon.overlayImages(warped[:,sh[1]//2,:], fixed[:,sh[1]//2,:])
    rcommon.overlayImages(warped[sh[0]//2,:,:], fixed[sh[0]//2,:,:])
    rcommon.overlayImages(warped[:,:,sh[2]//2], fixed[:,:,sh[2]//2])    

if __name__=="__main__":
    argc=len(sys.argv)
    if argc<2:
        print 'Task name expected:\n','segatlas\n','invert\n','npy2nifti\n','lattice\n'
        
        sys.exit(0)
    if(sys.argv[1]=='segatlas'):
        if argc<4:
            print "Two file names expected as arguments"
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
        lambdaParam=0.5
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
    print 'Unknown argument:',sys.argv[1]
