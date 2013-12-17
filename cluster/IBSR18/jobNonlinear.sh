#!/bin/bash
####################################################
# Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l pmem=4gb
#PBS -l nodes=1:ppn=1
#PBS -l walltime=04:00:00
#PBS -N Nonlinear
#PBS -M jomaroceguedag@gmail.com
export PATH="/opt/python/anaconda/bin:$PATH"
export PYTHONPATH="/opt/python/anaconda/lib/python2.7/site-packages:/export/opt/python/anaconda/lib/python2.7/site-packages:$PYTHONPATH:$HOME/code/registration"
export PATH=$HOME/opt:$PATH
export PYTHONPATH=$HOME/opt/lib/python2.7/site-packages:$PYTHONPATH
###################################
date
reference=$(ls reference)
target=$(ls target)
affine=$(ls affine)
python registrationNonlinear.py target/$target reference/$reference affine/$affine 100.0
date
exit 0
