# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:40:28 2013

@author: khayyam
"""
import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import nibabel as nib
import os
import tensorFieldUtils as tf
###############################################################
##########################  Common  ###########################
###############################################################
const_prefilter_map_coordinates=False


# def affine_registration(static, moving):
#     from nipy.io.files import nipy2nifti, nifti2nipy
#     from nipy.algorithms.registration import HistogramRegistration, resample
#     nipy_static = nifti2nipy(static)
#     nipy_moving = nifti2nipy(moving)
#     similarity = 'crl1' #'crl1' 'cc', 'mi', 'nmi', 'cr', 'slr'
#     interp = 'tri' #'pv', 'tri',
#     renormalize = True
#     optimizer = 'powell'
#     R = HistogramRegistration(nipy_static, nipy_moving, similarity=similarity,
#                           interp=interp, renormalize=renormalize)
#     T = R.optimize('affine', optimizer=optimizer)
#     warped= resample(nipy_moving, T, reference=nipy_static, interp_order=1)
#     warped = nipy2nifti(warped, strict=True)
#     return warped, T

# warped, affine_init = affine_registration(nib_static, nib_moving)

def getBaseFileName(fname):
    base=os.path.basename(fname)
    noExt=os.path.splitext(base)[0]
    while(noExt!=base):
        base=noExt
        noExt=os.path.splitext(base)[0]
    return noExt    
    

def getDistribution(img1, img2):
    sh=img1.shape
    dist=np.zeros((256,256))
    for i in range(sh[0]):
        for j in range(sh[1]):
            a=int(img1[i,j])
            b=int(img2[i,j])
            dist[a,b]+=1
    return dist

def drawLattice2D(nrows, ncols, delta):
    lattice=np.ndarray((1+(delta+1)*nrows, 1+(delta+1)*ncols), dtype=np.float64)
    lattice[...]=127
    for i in range(nrows+1):
        lattice[i*(delta+1), :]=0
    for j in range(ncols+1):
        lattice[:, j*(delta+1)]=0
    return lattice

def createDeformationField2D_type1(nrows, ncols, maxDistp):
    deff=np.ndarray((nrows, ncols, 2), dtype=np.float64)
    midCol=ncols//2
    midRow=nrows//2
    for i in range(nrows):
        deff[i,:,0]=maxDistp*np.sin(2*np.pi*(np.array(range(ncols), dtype=np.float64)-midCol)/ncols)
    for j in range(ncols):
        deff[:,j,1]=maxDistp*np.sin(2*np.pi*(np.array(range(nrows), dtype=np.float64)-midRow)/nrows)
    v=np.array(range(nrows), dtype=np.float64)-midRow
    h=np.array(range(ncols), dtype=np.float64)-midCol
    nrm=midRow**2+midCol**2
    p=np.exp(-(v[:,None]**2+h[None,:]**2)/(0.1*nrm))
    p=(p-p.min())/(p.max()-p.min())
    deff[:,:,0]*=p
    deff[:,:,1]*=p
    return deff

def createDeformationField2D_type2(nrows, ncols, maxDistp):
    deff=np.ndarray((nrows, ncols, 2), dtype=np.float64)
    midCol=ncols//2
    midRow=nrows//2
    for i in range(nrows):
        deff[i,:,0]=maxDistp*np.sin(2*np.pi*(np.array(range(ncols), dtype=np.float64)-midCol)/ncols)
        deff[i,:,1]=maxDistp*np.sin(2*np.pi*(np.array(range(ncols), dtype=np.float64)-midCol)/ncols)
    for j in range(ncols):
        deff[:,j,0]*=np.sin(2*np.pi*(np.array(range(nrows), dtype=np.float64)-midRow)/nrows)
        deff[:,j,1]*=np.sin(2*np.pi*(np.array(range(nrows), dtype=np.float64)-midRow)/nrows)
    return deff

def createDeformationField2D_type3(nrows, ncols, maxDistp):
    deff=np.ndarray((nrows, ncols, 2), dtype=np.float64)
    X0,X1=np.mgrid[0:nrows, 0:ncols]
    midCol=ncols//2
    midRow=nrows//2
    nn=np.sqrt(midCol*midCol+midRow*midRow)
    factor=maxDistp/nn
    deff[...,0]=(X1-midCol)*(-factor)
    deff[...,1]=(X0-midRow)*(factor)
    return deff

def drawLattice3D(dims, delta):
    dims=np.array(dims)
    nsquares=(dims-1)/(delta+1)
    lattice=np.zeros(shape=dims, dtype=np.float64)
    lattice[...]=127
    for i in range(nsquares[0]+1):
        lattice[i*(delta+1), :, :]=0
    for j in range(nsquares[1]+1):
        lattice[:, j*(delta+1), :]=0
    for k in range(nsquares[2]+1):
        lattice[:, :, k*(delta+1)]=0
    return lattice

def createDeformationField3D_type2(dims, maxDistp):
    deff=np.ndarray(dims+(3,), dtype=np.float64)
    dims=np.array(dims, dtype=np.int32)
    mid=dims//2
    factor=maxDistp**(1.0/3.0)
    for i in range(dims[0]):
        deff[i,:,:,0]=factor*np.sin(2*np.pi*(np.array(range(dims[0]), dtype=np.float64)-mid[0])/dims[0])
        deff[i,:,:,1]=factor*np.sin(2*np.pi*(np.array(range(dims[0]), dtype=np.float64)-mid[0])/dims[0])
        deff[i,:,:,2]=factor*np.sin(2*np.pi*(np.array(range(dims[0]), dtype=np.float64)-mid[0])/dims[0])
    for j in range(dims[1]):
        deff[:,j,:,0]*=factor*np.sin(2*np.pi*(np.array(range(dims[1]), dtype=np.float64)-mid[1])/dims[1])
        deff[:,j,:,1]*=factor*np.sin(2*np.pi*(np.array(range(dims[1]), dtype=np.float64)-mid[1])/dims[1])
        deff[:,j,:,2]*=factor*np.sin(2*np.pi*(np.array(range(dims[1]), dtype=np.float64)-mid[1])/dims[1])
    for k in range(dims[2]):
        deff[:,:,k,0]*=factor*np.sin(2*np.pi*(np.array(range(dims[2]), dtype=np.float64)-mid[2])/dims[2])
        deff[:,:,k,1]*=factor*np.sin(2*np.pi*(np.array(range(dims[2]), dtype=np.float64)-mid[2])/dims[2])
        deff[:,:,k,2]*=factor*np.sin(2*np.pi*(np.array(range(dims[2]), dtype=np.float64)-mid[2])/dims[2])
    return deff

    
def warpImage(image, displacement):
    sh=image.shape
    X0,X1=np.mgrid[0:sh[0], 0:sh[1]]
    warped=ndimage.map_coordinates(image, [X0+displacement[...,0], X1+displacement[...,1]], prefilter=const_prefilter_map_coordinates)
    return warped

def warpVolume(volume, displacement, volAffine=None, dispAffine=None):
    sh=volume.shape
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
    if(dispAffine!=None):
        X0, X1, X2=(dispAffine[0,0]*X0 + dispAffine[0,1]*X1 + dispAffine[0,2]*X2 + dispAffine[0,3]+displacement[...,0],
                    dispAffine[1,0]*X0 + dispAffine[1,1]*X1 + dispAffine[1,2]*X2 + dispAffine[1,3]+displacement[...,1],
                    dispAffine[2,0]*X0 + dispAffine[2,1]*X1 + dispAffine[2,2]*X2 + dispAffine[2,3]+displacement[...,2])
    else:
        X0+=displacement[...,0]
        X1+=displacement[...,1]
        X2+=displacement[...,2]
    if(volAffine!=None):
        volInverse=np.linalg.inv(volAffine)
        X0, X1, X2=(volInverse[0,0]*X0 + volInverse[0,1]*X1 + volInverse[0,2]*X2 + volInverse[0,3],
                    volInverse[1,0]*X0 + volInverse[1,1]*X1 + volInverse[1,2]*X2 + volInverse[1,3],
                    volInverse[2,0]*X0 + volInverse[2,1]*X1 + volInverse[2,2]*X2 + volInverse[2,3])
    warped=ndimage.map_coordinates(volume, [X0, X1, X2], prefilter=const_prefilter_map_coordinates)
    return warped

def plotDeformationField(d):
    plt.figure()
    plt.quiver(d[...,1], d[...,0])

def plotDeformedLattice(d, delta=10):
    nrows=d.shape[0]
    ncols=d.shape[1]
    lattice=drawLattice2D((nrows+delta)/(delta+1), (ncols+delta)/(delta+1), delta)
    lattice=lattice[0:nrows,0:ncols]
    gtLattice=warpImage(lattice, d)
    plt.figure()
    plt.imshow(gtLattice, cmap=plt.cm.gray)
    plt.title('Deformed lattice')
    return gtLattice

def plotDeformationFields(dList):
    n=len(dList)
    plt.figure()
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.quiver(dList[i][...,1], dList[i][...,0])

def plotOrthogonalField(sh, b):
    center=(np.array(sh)-1)/2.0
    C,R=sp.meshgrid(np.array(range(sh[1]), dtype=np.float64), np.array(range(sh[0]), dtype=np.float64))
    R=R-center[0]+b[0]
    C=C-center[1]+b[1]
    plt.figure()
    plt.quiver(R, -C)

def plotPyramids(L, R):
    n=len(L)
    plt.figure()
    for i in range(n):
        plt.subplot(2,n,i+1)
        plt.imshow(L[i], cmap = plt.cm.gray)
        plt.title('Level: '+str(i))
        plt.subplot(2,n,n+i+1)
        plt.imshow(R[i], cmap = plt.cm.gray)
        plt.title('Level: '+str(i))

def renormalize_image(image):
    m=np.min(image)
    M=np.max(image)
    if(M-m<1e-8):
        return image
    return 127.0*(image-m)/(M-m)

def plotOverlaidPyramids(L, R):
    n=len(L)
    plt.figure()
    for i in range(n):
        plt.subplot(1,n,i+1)
        colorImage=np.zeros(shape=(L[i].shape)+(3,), dtype=np.int8)
        ll=renormalize_image(L[i]).astype(np.int8)
        rr=renormalize_image(R[i]).astype(np.int8)
        colorImage[...,0]=ll*(ll>ll[0,0])
        colorImage[...,1]=rr*(rr>rr[0,0])
        plt.imshow(colorImage)
        plt.title('Level: '+str(i))
        
def plotOverlaidPyramids3DCoronal(L, R):
    n=len(L)
    plt.figure()
    for i in range(n):
        sh=L[i].shape
        plt.subplot(1,n,i+1)
        colorImage=np.zeros(shape=(sh[0], sh[2], 3), dtype=np.int8)
        ll=renormalize_image(L[i][:,sh[1]//2,:]).astype(np.int8)
        rr=renormalize_image(R[i][:,sh[1]//2,:]).astype(np.int8)
        colorImage[...,0]=ll*(ll>ll[0,0])
        colorImage[...,1]=rr*(rr>rr[0,0])
        plt.imshow(colorImage)
        plt.title('Level: '+str(i))

def getRotationMatrix(angles):
    ca=np.cos(angles[0])
    cb=np.cos(angles[1])
    cg=np.cos(angles[2])
    sa=np.sin(angles[0])
    sb=np.sin(angles[1])
    sg=np.sin(angles[2])
    return np.array([[cb*cg,-ca*sg+sa*sb*cg,sa*sg+ca*sb*cg],[cb*sg,ca*cg+sa*sb*sg,-sa*cg+ca*sb*sg],[-sb,sa*cb,ca*cb]])

def getRotationMatrix2D(angle):
    c=np.cos(angle)
    s=np.sin(angle)
    return np.array([[c, -s],[s, c]])


def applyRigidTransformation3D(image, beta):
    sh=image.shape
    center=(np.array(sh)-1)/2.0
    X0,X1,X2=np.mgrid[0:sh[0], 0:sh[1], 0:sh[2]]
    X0=X0-center[0]
    X1=X1-center[1]
    X2=X2-center[2]
    R=getRotationMatrix(beta[0:3])
    X0new,X1new,X2new=(R[0,0]*X0 + R[0,1]*X1 + R[0,2]*X2 + center[0] + beta[3], 
                       R[1,0]*X0 + R[1,1]*X1 + R[1,2]*X2 + center[1] + beta[4], 
                       R[2,0]*X0 + R[2,1]*X1 + R[2,2]*X2 + center[2] + beta[5])
    return ndimage.map_coordinates(image, [X0new, X1new, X2new], prefilter=const_prefilter_map_coordinates)

def pyramid_gaussian_3D(image, max_layer, mask=None):
    yield image.copy().astype(np.float64)
    for i in range(max_layer):
        newImage=sp.ndimage.filters.gaussian_filter(image, 2.0/3.0)[::2,::2,::2].copy()
        if(mask!=None):
            mask=mask[::2,::2,::2]
            newImage*=mask
        image=newImage.copy()
        yield newImage

#def pyramid_gaussian_3D(image, max_layer, mask=None):
#    yield image.copy().astype(np.float64)
#    for i in range(max_layer):
#        newImage=np.array(tf.downsample_scalar_field3D(image))
#        image=newImage
#        yield newImage

def pyramid_gaussian_2D(image, max_layer, mask=None):
    yield image.copy().astype(np.float64)
    for i in range(max_layer):
        newImage=np.empty(shape=((image.shape[0]+1)//2, (image.shape[1]+1)//2), dtype=np.float64)
        newImage[...]=sp.ndimage.filters.gaussian_filter(image, 2.0/3.0)[::2,::2]
        if(mask!=None):
            mask=mask[::2,::2]
            newImage*=mask
        image=newImage
        yield newImage

#def pyramid_gaussian_2D(image, max_layer, mask=None):
#    yield image.copy().astype(np.float64)
#    for i in range(max_layer):
#        newImage=np.array(tf.downsample_scalar_field(image))
#        image=newImage
#        yield newImage


def overlayImages(img0, img1, createFig=True):
    colorImage=np.zeros(shape=(img0.shape)+(3,), dtype=np.int8)
    colorImage[...,0]=renormalize_image(img0)
    colorImage[...,1]=renormalize_image(img1)
    fig=None
    if(createFig):
        fig=plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img0, cmap=plt.cm.gray)
    plt.title('Img0 (red)')
    plt.subplot(1,3,2)
    plt.imshow(colorImage)
    plt.title('Overlay')
    plt.subplot(1,3,3)
    plt.imshow(img1, cmap=plt.cm.gray)
    plt.title('Img1 (green)')
    return fig

def testOverlayImages_raw():
    leftName='data/t2/t2_icbm_normal_1mm_pn0_rf0.rawb'
    rightName='data/t1/t1_icbm_normal_1mm_pn0_rf0.rawb'
    ns=181
    nr=217
    nc=181
    left=np.fromfile(leftName, dtype=np.ubyte).reshape(ns,nr,nc)
    left=left.astype(np.float64)
    right=np.fromfile(rightName, dtype=np.ubyte).reshape(ns,nr,nc)
    right=right.astype(np.float64)
    overlayImages(left[90], right[90])

def getColor(label):
    r=label%2
    g=(label/2)%2
    b=(label/4)%2
    return np.array([r,g,b])    

def testOverlayImages_nii():
    leftName='data/t1/IBSR18/IBSR_01/IBSR_01_ana_strip.nii.gz'
    rightName='data/t1/IBSR18/IBSR_02/IBSR_02_ana_strip.nii.gz'
    nib_left = nib.load(leftName)
    nib_right = nib.load(rightName)
    left=nib_left.get_data().astype(np.double)
    right=nib_right.get_data().astype(np.double)
    left=np.copy(left, order='C')
    right=np.copy(right, order='C')
    sl=np.array(left.shape)//2
    sr=np.array(right.shape)//2
    overlayImages(left[sl[0],:,:,0], right[sr[0],:,:,0])
    overlayImages(left[:,sl[1],:,0], right[:,sr[1],:,0])
    overlayImages(left[:,:,sl[2],0], right[:,:,sr[2],0])

def plotDiffeomorphism(GT, GTinv, GTres, titlePrefix, delta=10):
    nrows=GT.shape[0]
    ncols=GT.shape[1]
    X1,X0=np.mgrid[0:GT.shape[0], 0:GT.shape[1]]
    lattice=drawLattice2D((nrows+delta)/(delta+1), (ncols+delta)/(delta+1), delta)
    lattice=lattice[0:nrows,0:ncols]
    gtLattice=warpImage(lattice, np.array(GT))
    gtInvLattice=warpImage(lattice, np.array(GTinv))
    gtResidual=warpImage(lattice, np.array(GTres))
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(gtLattice, cmap=plt.cm.gray)
    plt.title(titlePrefix+'[Deformation]')
    plt.subplot(2, 3, 2)
    plt.imshow(gtInvLattice, cmap=plt.cm.gray)
    plt.title(titlePrefix+'[Inverse]')
    plt.subplot(2, 3, 3)
    plt.imshow(gtResidual, cmap=plt.cm.gray)
    plt.title(titlePrefix+'[residual]')
    #plot jacobians and residual norm
    detJacobian=computeJacobianField(GT)
    plt.subplot(2, 3, 4)
    plt.imshow(detJacobian, cmap=plt.cm.gray)
    CS=plt.contour(X0,X1,detJacobian,levels=[0.0], colors='r')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('det(J(d))')    
    detJacobianInverse=computeJacobianField(GTinv)
    plt.subplot(2, 3, 5)
    plt.imshow(detJacobianInverse, cmap=plt.cm.gray)
    CS=plt.contour(X0,X1,detJacobianInverse,levels=[0.0], colors='r')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('det(J(d^-1))')    
    nrm=np.sqrt(np.sum(np.array(GTres)**2,2))
    plt.subplot(2, 3, 6)
    plt.imshow(nrm, cmap=plt.cm.gray)
    plt.title('||residual||_2')    
    g00, g01=sp.gradient(GTinv[...,0])
    g10, g11=sp.gradient(GTinv[...,1])
    #priorEnergy=g00**2+g01**2+g10**2+g11**2
    return [gtLattice, gtInvLattice, gtResidual, detJacobian]

def plot_2d_diffeomorphic_map(mapping, delta=10, fname = None):
    #Create a grid on the moving domain
    nrows_moving = mapping.forward.shape[0]
    ncols_moving = mapping.forward.shape[1]
    X1,X0=np.mgrid[0:nrows_moving, 0:ncols_moving]
    lattice_moving=drawLattice2D((nrows_moving+delta)/(delta+1), 
                                 (ncols_moving+delta)/(delta+1), delta)
    lattice_moving=lattice_moving[0:nrows_moving, 0:ncols_moving]
    #Warp in the forward direction (since the lattice is in the moving domain)
    warped_forward = mapping.transform(lattice_moving)

    #Create a grid on the static domain
    nrows_static = mapping.backward.shape[0]
    ncols_static = mapping.backward.shape[1]
    X1,X0=np.mgrid[0:nrows_static, 0:ncols_static]
    lattice_static=drawLattice2D((nrows_static+delta)/(delta+1), 
                                 (ncols_static+delta)/(delta+1), delta)
    lattice_static=lattice_static[0:nrows_static, 0:ncols_static]
    #Warp in the backward direction (since the lattice is in the static domain)
    warped_backward = mapping.transform_inverse(lattice_static)

    #Now plot the grids
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(warped_forward, cmap=plt.cm.gray)
    plt.title('Direct transform')
    plt.subplot(1, 3, 2)
    plt.imshow(lattice_moving, cmap=plt.cm.gray)
    plt.title('Original grid')
    plt.subplot(1, 3, 3)
    plt.imshow(warped_backward, cmap=plt.cm.gray)
    plt.title('Inverse transform')
    if fname is not None:
      from time import sleep
      sleep(1)
      plt.savefig(fname, bbox_inches='tight')


def computeJacobianField(displacement):
    g00,g01=sp.gradient(displacement[...,0])
    g10,g11=sp.gradient(displacement[...,1])
    return (1+g00)*(1+g11)-g10*g01

def saveDeformedLattice3D(dname, oname='deformed_lattice.nii.gz'):
    '''
        saveDeformedLattice3D('displacement_templateT1ToIBSR01T1_diff.npy')
        saveDeformedLattice3D('displacement_templateT1ToIBSR01T1_diffMulti.npy')
    '''
    print 'Loading displacement...'
    displacement=np.load(dname)
    minVal, maxVal=tf.get_displacement_range(displacement, None)
    sh=np.array([np.ceil(maxVal[0]),np.ceil(maxVal[1]),np.ceil(maxVal[2])], dtype=np.int32)
    print sh.dtype
    print sh
    L=np.array(drawLattice3D(sh, 10))
    print 'Warping lattice...'
    warped=np.array(tf.warp_volume(L, displacement, np.eye(4))).astype(np.int16)
    print 'Transforming to Nifti...'
    img=nib.Nifti1Image(warped, np.eye(4))
    print 'Saving warped lattice as:',oname
    img.to_filename(oname)
    print 'done.'

def readAntsAffine(fname):
    '''
    readAntsAffine('IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt')
    '''
    try:
        with open(fname) as f:
            lines=[line.strip() for line in f.readlines()]
    except IOError:
        print 'Can not open file: ', fname
        return
    if not (lines[0]=="#Insight Transform File V1.0"):
        print 'Unknown file format'
        return
    if lines[1]!="#Transform 0":
        print 'Unknown transformation type'
        return
    A=np.zeros((3,3))
    b=np.zeros((3,))
    c=np.zeros((3,))
    for line in lines[2:]:
        data=line.split()
        if data[0]=='Transform:':
            if data[1]!='MatrixOffsetTransformBase_double_3_3':
                print 'Unknown transformation type'
                return
        elif data[0]=='Parameters:':
            parameters=np.array([float(s) for s in data[1:]], dtype=np.float64)
            A=parameters[:9].reshape((3,3))
            b=parameters[9:]
        elif data[0]=='FixedParameters:':
            c=np.array([float(s) for s in data[1:]], dtype=np.float64)
    T=np.ndarray(shape=(4,4), dtype=np.float64)
    T[:3,:3]=A[...]
    T[3,:]=0
    T[3,3]=1
    T[:3,3]=b+c-A.dot(c)
    ############This conversion is necessary for compatibility between itk and nibabel#########
    conversion=np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    T=conversion.dot(T.dot(conversion))
    ###########################################################################################
    return T

def create_3d_grid(min_bounds=[0, 0, 0], max_bounds=[128, 256, 256], nlines=[20, 20, 20], npoints=50):
    r"""
    Creates a set of streamlines forming a regular 3D grid
    """
    grid = []
    #Create the lines along the first axis
    for i in range(nlines[1]):
        for j in range(nlines[2]):
            z0, z1 = min_bounds[0], max_bounds[0]
            y = min_bounds[1] + i * (max_bounds[1] - min_bounds[1])/(nlines[1]-1)
            x = min_bounds[2] + j * (max_bounds[2] - min_bounds[2])/(nlines[2]-1)
            t = np.linspace(z0, z1, npoints)
            streamline = np.vstack((t, np.zeros_like(t)+y, np.zeros_like(t)+x)).T
            grid.append(streamline)
    #Create the lines along the second axis
    for k in range(nlines[1]):
        for j in range(nlines[2]):
            y0, y1 = min_bounds[1], max_bounds[1]
            z = min_bounds[0] + k * (max_bounds[0] - min_bounds[0])/(nlines[0]-1)
            x = min_bounds[2] + j * (max_bounds[2] - min_bounds[2])/(nlines[2]-1)
            t = np.linspace(y0, y1, npoints)
            streamline = np.vstack((np.zeros_like(t)+z, t, np.zeros_like(t)+x)).T
            grid.append(streamline)
    #Create the lines along the third axis
    for k in range(nlines[0]):
        for i in range(nlines[1]):
            x0, x1 = min_bounds[2], max_bounds[2]
            z = min_bounds[0] + k * (max_bounds[0] - min_bounds[0])/(nlines[0]-1)
            y = min_bounds[1] + i * (max_bounds[1] - min_bounds[1])/(nlines[1]-1)
            t = np.linspace(x0, x1, npoints)
            streamline = np.vstack((np.zeros_like(t)+z, np.zeros_like(t)+y, t)).T
            grid.append(streamline)
    return grid

def create_2d_grid(min_bounds=[0, 0], max_bounds=[256, 256], nlines=[20, 20], npoints=50):
    r"""
    Creates a set of streamlines forming a regular 2D grid
    """
    grid = []
    #Create the lines along the first axis
    for j in range(nlines[1]):
        y0, y1 = min_bounds[0], max_bounds[0]
        x = min_bounds[1] + j * (max_bounds[1] - min_bounds[1])/(nlines[1]-1)
        t = np.linspace(y0, y1, npoints)
        streamline = np.vstack((t, np.zeros_like(t)+x)).T
        grid.append(streamline)
    #Create the lines along the second axis
    for i in range(nlines[0]):
        x0, x1 = min_bounds[1], max_bounds[1]
        y = min_bounds[0] + i * (max_bounds[0] - min_bounds[0])/(nlines[0]-1)
        t = np.linspace(x0, x1, npoints)
        streamline = np.vstack((np.zeros_like(t)+y, t)).T
        grid.append(streamline)
    return grid

def plot_interactive_2d_grid(grid):
    extended_grid = []
    for line in grid:
        extended_line = np.zeros(shape = (line.shape[0], 3))
        extended_line[...,1:3] = line
        extended_grid.append(extended_line) 
    ren = fvtk.ren()
    ren.SetBackground(*fvtk.colors.white)
    grid_actor = fvtk.streamtube(extended_grid, fvtk.colors.red, linewidth=0.3)
    fvtk.add(ren, grid_actor)
    fvtk.camera(ren, pos=(0, 0, 0), focal=(30, 0, 0))
    fvtk.show(ren)

def warp_all_streamlines(streamlines, mapping):
    import vector_fields as vf
    warped = []
    for stremline in streamlines:
        line = streamline.astype(floating)
        wline = vf.warp_2d_stream_line(line, mapping.forward, 
                                       mapping.affine_pre, mapping.affine_post)
        warped.append(wline)
    return warped

def plot_middle_slices(V, fname=None):
    sh=V.shape
    axial = renormalize_image(V[sh[0]//2, :, :]).astype(np.int8)
    coronal = renormalize_image(V[:, sh[1]//2, :]).astype(np.int8)
    sagital = renormalize_image(V[:, :, sh[2]//2]).astype(np.int8)

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(axial, cmap = plt.cm.gray)
    plt.title('Axial')
    plt.subplot(1,3,2)
    plt.imshow(coronal, cmap = plt.cm.gray)
    plt.title('Coronal')
    plt.subplot(1,3,3)
    plt.imshow(sagital, cmap = plt.cm.gray)
    plt.title('Sagital')
    if fname is not None:
        from time import sleep
        sleep(1)
        plt.savefig(fname, bbox_inches='tight')
