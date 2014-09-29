#!/bin/bash
####################################################
# Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l mem=4GB
#PBS -l pmem=4GB
#PBS -l vmem=4GB
#PBS -l nodes=1:ppn=1
#PBS -l walltime=03:00:00
#PBS -N FullANTS
export PATH="/opt/python/anaconda/bin:$PATH"
export PYTHONPATH="/opt/python/anaconda/lib/python2.7/site-packages:/export/opt/python/anaconda/lib/python2.7/site-packages:$PYTHONPATH:$HOME/code/registration"
export PATH=$HOME/opt:$PATH
export PYTHONPATH=$HOME/opt/dipy:$HOME/opt/lib/python2.7/site-packages:$PYTHONPATH
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
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
affine0="${targetbase}_${referencebase}0Affine.txt"
affinePrecomputed="../affine/${affine}"
if [ -r $affinePrecomputed ]; then
    cp $affinePrecomputed .
fi

op="${targetbase}_${referencebase}"

if ! [ -r $affine ]; then
    exe="antsRegistration -d 3 -r [ reference/$reference, target/$target, 1 ] \
                      -m mattes[ reference/$reference, target/$target, 1 , 32, regular, 0.3 ] \
                      -t translation[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 6x4x2 -l 1 \
                      -m mattes[ reference/$reference, target/$target, 1 , 32, regular, 0.3 ] \
                      -t rigid[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 3x2x1 -l 1 \
                      -m mattes[ reference/$reference, target/$target, 1 , 32, regular, 0.3 ] \
                      -t affine[ 0.1 ] \
                      -c [ 10000x111110x11110,1.e-8,20 ] \
                      -s 4x2x1vox \
                      -f 3x2x1 -l 1 \
                      -o [${op}]"
    echo " $exe "
    $exe
    mv $affine0 $affine
else
    echo "Affine mapping found ($affine). Skipping affine registration."
fi
#Diffeomorphic registration
deformationField=${targetbase}_${referencebase}Warp.nii.gz
if [ -r $deformationField ]; then
    echo "Deformation found. Registration skipped."
else
    exe="antsRegistration -d 3 -r $affine \
                      -m mattes[reference/$reference, target/$target, 1 , 32 ] \
                      -t syn[ .25, 3, 0 ] \
                      -c [ 100x100x25,-0.01,5 ] \
                      -s 1x0.5x0vox \
                      -f 4x2x1 -l 1 -u 0 -z 1 \
                      --float \
                      -o [${op}, warpedDiff_${op}.nii.gz, warpedDiff_${op}.nii.gz]"
    echo " $exe "
    $exe
fi

date

for towarp in $( ls warp ); do
    towarpbase="${towarp%.*}"
    towarpbase="${towarpbase%.*}"
    oname=warpedDiff_${towarpbase}_${referencebase}.nii.gz
    deformationField=${targetbase}_${referencebase}1Warp.nii.gz
    antsApplyTransforms -d 3 -i warp/$towarp -o $oname -r reference/$reference -n NearestNeighbor --float -t $deformationField -t $affine
done

python jaccard.py

