#!/bin/bash
####################################################
# Author: Omar Ocegueda (omar@cimat.mx)
#PBS -l pmem=4gb
#PBS -l nodes=1:ppn=1
#PBS -l walltime=04:00:00
#PBS -N Diffeomorphic
#PBS -M jomaroceguedag@gmail.com
export PATH="/opt/python/anaconda/bin:$PATH"
export PYTHONPATH=/opt/python/anaconda/lib/python2.7/site-packages:/home/omar/code/registration
###################################
date
reference=$(ls reference)
target=$(ls target)
python registrationDiffeomorphic.py target/$target reference/$reference 10.0
date
exit 0
