#!/bin/bash
####################################################
# Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=4gb
#PBS -l nodes=1:ppn=4
#PBS -l walltime=03:00:00
#PBS -N FullANTS
#PBS -M omar@cimat.com
export PATH="/opt/python/anaconda/bin:$PATH"
export PYTHONPATH="/opt/python/anaconda/lib/python2.7/site-packages:/export/opt/python/anaconda/lib/python2.7/site-packages:$PYTHONPATH:$HOME/code/registration"
export PATH=$HOME/opt:$PATH
export PYTHONPATH=$HOME/opt/lib/python2.7/site-packages:$PYTHONPATH
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
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
deformationField=${targetbase}_${referencebase}Warp.nii.gz
if [ -r $deformationField ]; then
    echo "Deformation found. Registration skipped."
else
    exe="ANTS 3 -m  MI[reference/$reference,target/$target,1,3] -t SyN[0.5] -a ${affine} -r Gauss[2,0] -o ${targetbase}_${referencebase} -i 1x1x1 --continue-affine false"
    echo " $exe "
    $exe
fi

for towarp in $( ls warp ); do
    towarpbase="${towarp%.*}"
    towarpbase="${towarpbase%.*}"
    oname=${towarpbase}_${referencebase}.nii.gz
    deformationField=${targetbase}_${referencebase}Warp.nii.gz
    WarpImageMultiTransform 3 warp/$towarp $oname $deformationField $affine -R reference/$reference --use-NN
done
date

