#! /usr/bin/env python
"""
This script investigates the use of affine registration in registering 3D
DWI volumes to the S0 volume as done in standard eddy current correction.
"""

import time
import numpy as np
import nibabel as nib
from nipy.algorithms.registration import HistogramRegistration, resample
from nipy.io.files import nipy2nifti, nifti2nipy

from dipy.segment.mask import median_otsu

from dipy.fixes import argparse as arg

parser = arg.ArgumentParser(description='Affine registration')

parser.add_argument('in_file', action='store', metavar='in_file',
                    help='Nifti1 image (*.nii or *.nii.gz) or other formats supported by Nibabel')

parser.add_argument('reference', action='store', metavar='reference',
                    help='Nifti1 image (*.nii or *.nii.gz) or other formats supported by Nibabel')

parser.add_argument('out_file', action='store', metavar='out_file',
                    help='Nifti1 image (*.nii or *.nii.gz) or other formats supported by Nibabel')

parser.add_argument('--similarity', action='store', metavar='String',
                    help="Cost-function for assessing image similarity. If a string, one of 'cc': correlation coefficient, 'cr': correlation ratio, 'crl1': L1-norm based correlation ratio, 'mi': mutual information, 'nmi': normalized mutual information, 'slr': supervised log-likelihood ratio. If a callable, it should take a two-dimensional array representing the image joint histogram as an input and return a float.",
                    default='crl1')

parser.add_argument('--interp', action='store', metavar='String',
                   help="'Interpolation method.One of 'pv': Partial volume, 'tri':Trilinear, 'rand': Random interpolation'",
                   default='pv')

params = parser.parse_args()


if __name__ == '__main__':

    fmoving = params.in_file
    fstatic = params.reference
    fmoved = params.out_file

    print(fmoving + ' --> ' + fstatic)

    static = nifti2nipy(nib.load(fstatic))
    moving = nifti2nipy(nib.load(fmoving))

    similarity = params.similarity #'crl1' 'cc', 'mi', 'nmi', 'cr', 'slr'
    interp = params.interp #'pv', 'tri',
    renormalize = True
    optimizer = 'powell'

    print('Setting up registration...')
    tic = time.time()
    R = HistogramRegistration(moving, static, similarity=similarity,
                              interp=interp, renormalize=renormalize)

    T = R.optimize('affine', optimizer=optimizer)
    toc = time.time()
    print('Registration time: %f sec' % (toc - tic))

    moving_new = resample(moving, T.inv(), reference=static)

    nib.save(nipy2nifti(moving_new, strict=True), fmoved)