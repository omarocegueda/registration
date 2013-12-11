#!/bin/bash
####################################################
# Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l pmem=4gb
#PBS -l nodes=1:ppn=1
#PBS -l walltime=01:30:00
#PBS -N FullRegistration
#PBS -M jomaroceguedag@gmail.com
export PATH="/opt/python/anaconda/bin:$PATH"
export PYTHONPATH=/opt/python/anaconda/lib/python2.7/site-packages:/home/omar/code/registration
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
ANTS 3 -m MI[reference/$reference, target/$target, 1, 32] -i 0 -o ${targetbase}_${referencebase}
#Diffeomorphic registration
affine="${targetbase}_${referencebase}Affine.txt"
python registrationDiffeomorphic.py target/$target reference/$reference $affine 100.0
date
