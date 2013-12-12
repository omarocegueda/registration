#!/bin/bash
#######################################################################
# Author: Omar Ocegueda (omar@cimat.mx)
# This script creates a directory containing the necessary data 
# (a link to the data) to register two IBSR brains with a diffeomorphic
# non-linear transformation per node in the cluster. 
# If 's' is given as first parameter, it prepares the registration jobs.
# It receives as input two text files, say movingNames.txt and 
# fixedNames.txt containing the absolute names of the moving and fixed 
# files to be registered, respectively (they must have the same number 
# of lines).
# If 'c' is given as parameter, it 'cleans' the working directory
# by force-removing all the subdirectories.
# If 'o' is given as parameter, it collects the registration results
# and puts it into a folder named 'results'.
# If 'u' is given as parameter, it submits the jobs to the cluster.
#
# Usage: ./splitDiffeomorphicMR.sh [option][fname]
# Example (clean): ./splitDiffeomorphicMR.sh c
# Example (split): ./splitDiffeomorphicMR.sh s namesMoving.txt namesFixed.txt
# Example (submit): ./splitDiffeomorphicMR.sh u
# Example (collect):./splitDiffeomorphicMR.sh o
#
#######################################################################
for dir in $(ls -d [0-9]*/); do
    lines=(`cat ${dir}/*.e*`)
    nlines=${#lines[@]}
    if [ $nlines -ne 0 ]; then
        rm ${dir}/*.e*
        rm ${dir}/*.o*
        cd ${dir}
        qsub job*.sh -d .
        cd ..
    fi    
done

