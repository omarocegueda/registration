#!/bin/bash
####################################################
# Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l nodes=1:ppn=1
#PBS -l walltime=01:00:00
#PBS -N Affine
export PATH=$PATH:$HOME/bin:$HOME/opt/bin:/opt/python/anaconda/bin:/home/omar/opt/ANTs/bin/bin:$PATH
###################################
date
reference=$(ls reference)
target=$(ls target)
extension="${target##*.}"
targetbase="${target%.*}"
targetbase="${targetbase%.*}"
referencebase="${reference%.*}"
referencebase="${referencebase%.*}"
ANTS 3 -m MI[reference/$reference, target/$target, 1, 32] -i 0 -o ${targetbase}_${referencebase}
WarpImageMultiTransform 3 target/$target ${targetbase}_$reference -R reference/$reference ${targetbase}_${referencebase}Affine.txt
date
exit 0
