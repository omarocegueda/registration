#!/bin/bash
nthreads=1
####################################################
# Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=2gb
#PBS -l nodes=1:ppn=1
#PBS -l walltime=04:00:00
#PBS -N FullFFD
export PATH=$HOME/opt:$PATH
###################################
date
reference=$(ls reference)
target=$(ls target)
extension="${target##*.}"
targetbase="${target%.*}"
targetbase="${targetbase%.*}"
referencebase="${reference%.*}"
referencebase="${referencebase%.*}"
#Diffeomorphic registration
exe0="cmtk registration --echo --threads ${nthreads} --verbose-level 2 --auto-multi-levels 4 --nmi --dofs 6,9 -o affine_directory reference/${reference} target/${target}"
$exe0
deformationField=${targetbase}_${referencebase}.xform
exe1="cmtk warp --echo --threads ${nthreads} --verbose-level 2 --force-outside-value 0 --grid-spacing 40 --coarsest 4 --relax-to-unfold --refine 3 --jacobian-weight 1e-4 --ncc -e 16 -a 0.125 -o ${deformationField} affine_directory"
$exe1
for towarp in $( ls warp ); do
    towarpbase="${towarp%.*}"
    towarpbase="${towarpbase%.*}"
    oname=${towarpbase}_${referencebase}.nii.gz
    exe2="cmtk --echo --threads ${nthreads} --verbose-level 2 reformatx --nn -o $oname --floating warp/$towarp reference/${reference} ${deformationField}"
    $exe2
done
date

