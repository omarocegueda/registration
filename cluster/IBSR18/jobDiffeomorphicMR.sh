#!/bin/bash
####################################################
# Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:00:30
#PBS -N Diffeomorphic
export PYTHONPATH=/opt/python/anaconda/lib/python2.7/site-packages:/home/omar/code/registration
###################################
date
reference=$(ls reference)
target=$(ls target)
python registrationDiffeomorphic.py target/$target reference/$reference 10.0
date
exit 0
