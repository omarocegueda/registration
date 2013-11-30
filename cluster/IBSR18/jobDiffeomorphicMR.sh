#!/bin/bash
####################################################
# Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:01:00
#PBS -N Diffeomorphic
export DATA_DIR=/home/omar/code/registration/IBSR_nifti_stripped
export CODE_DIR=/home/omar/code/registration
export WORK_DIR=/home/omar/experiments/registration
mkdir -p $WORK_DIR
###################################
date
ls -l *.gz
date
exit 0
