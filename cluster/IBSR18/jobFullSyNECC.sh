#!/bin/bash
####################################################
# Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l pmem=3gb
#PBS -l nodes=1:ppn=1
#PBS -l walltime=04:00:00
#PBS -N FullSyNECC
#PBS -M omar@cimat.com
export PATH="/opt/python/anaconda/bin:$PATH"
export PYTHONPATH="/opt/python/anaconda/lib/python2.7/site-packages:/export/opt/python/anaconda/lib/python2.7/site-packages:$PYTHONPATH:$HOME/code/registration"
export PATH=$HOME/opt:$PATH
export PYTHONPATH=$HOME/opt/dipy:$HOME/opt/lib/python2.7/site-packages:$PYTHONPATH
###################################
date
reference=$(ls reference)
target=$(ls target)
extension="${target##*.}"
targetbase="${target%.*}"
targetbase="${targetbase%.*}"
referencebase="${reference%.*}"
referencebase="${referencebase%.*}"
#Affine registration using Mutual information with ANTS
affine="${targetbase}_${referencebase}Affine.txt"
affinePrecomputed="../affine/${affine}"
if [ -r $affinePrecomputed ]; then
    cp $affinePrecomputed .
fi
if ! [ -r $affine ]; then
    ANTS 3 -m MI[reference/$reference, target/$target, 1, 32] -i 0 -o ${targetbase}_${referencebase}
else
    echo "Affine mapping found ($affine). Skipping affine registration."
fi
#Diffeomorphic registration
python dipyreg.py target/$target reference/$reference $affine warp --metric=ECC[2.0,4,256] --iter=25,100,100 --step_length=0.25
date
