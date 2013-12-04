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
pythonScriptName="/home/omar/code/registration/registrationDiffeomorphic.py"
#############################No parameters#############################
if [ -z "$1" ]; then
    echo Please specify an action: c "(clean)", s "(split)", u "(submit)", o "(collect)"
    exit 0
fi
#############################Clean#####################################
if [ "$1" == "c" ]; then
    if [ ! -d "results" ]; then
        read -p "It seems like you have not collected the results yet. Clean anyway? (y/n)" -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
    for name in $(ls -d [0-9]*/); do
        $(rm -R -f $name)
    done
    exit 0
fi
#############################Split####################################
if [ "$1" == "s" ]; then
    if [[ -z "$2" || -z "$3" ]]; then
        echo Please specify two text file names: moving names and fixed names \(they must have the same number of lines\)
        exit 0
    fi
    movingNames=(`cat "$2"`)
    lenMoving=${#movingNames[@]}
    fixedNames=(`cat "$3"`)
    lenFixed=${#fixedNames[@]}
    if [ $lenMoving -ne $lenFixed ]; then
        echo Moving names and Fixed names must have the same number of file names. Moving has $lenMoving, fixed has $lenFixed
        exit 0
    fi
    for index in ${!movingNames[*]}; do
        folderIndex=$[$index +1 ]
        stri="$folderIndex"
        if [[ $folderIndex -lt 10 ]]; then
            stri="0$folderIndex"
        fi
        mkdir -p "$stri"/target
        mkdir -p "$stri"/reference
        ln "${movingNames[$index]}" $stri/target
        ln "${fixedNames[$index]}" $stri/reference
        cp jobDiffeomorphicMR.sh $stri
        cp $pythonScriptName $stri
    done
    exit 0
fi
############################Submit###################################
if [ "$1" == "u" ]; then
    for name in $(ls -d */); do
        echo Submitting: \'$name\'
        cd $name
        qsub jobDiffeomorphicMR.sh -d .
        cd ..
    done
    exit 0
fi
############################Collect##################################
if [ "$1" == "o" ]; then
    mkdir -p results
    for dir in $(ls -d [0-9]*/); do
        mv $dir/*.npy results
        mv $dir/*.nii.gz results
    done
    exit 0    
fi
############################Unknown##################################
echo Unknown option \'$1\'. The available options are "(c)"lean, "(s)"plit, s"(u)"bmit, c"(o)"llect.
exit 0

