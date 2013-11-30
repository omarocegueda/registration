#!/bin/bash
#######################################################################
# Author: Omar Ocegueda (omar@cimat.mx)
# This script creates a directory containing the necessary data 
# (a link to the data) and executables to register one input file 
# per node in the cluster. It receives as input a text file containing
# the absolute path of all the desired IBSR files to be registered.
# If 'c' is given as parameter, it 'cleans' the working directory
# by force-removing all the subdirectories.
#
# Usage: ./splitDiffeomorphicMR.sh [option][fname]
# Example (clean): ./splitDiffeomorphicMR.sh c
# Example (split): ./splitDiffeomorphicMR.sh s names.txt
# Example (submit): ./splitDiffeomorphicMR.sh u
#
# Note: please edit the CODE_DIR variable to specify the location of
# the registration executables
#######################################################################
CODE_DIR="/home/omar/code/registration"
#############################No parameters#############################
if [ -z "$1" ]; then
    echo Please specify an action: c "(clean)", s "(split)", u "(submit)"
    exit 0
fi
#############################Clean#####################################
if [ "$1" == "c" ]; then
    for name in $(ls -d */); do
        $(rm -R -f $name)
    done
    exit 0
fi
#############################Split####################################
if [ "$1" == "s" ]; then
    namesFile=$2
    if [ -z "$namesFile" ]; then
        echo Please specify the name of a text file containing the names of the IBSR input files to split
        exit 0
    fi
    i=1
    for name in $(cat $namesFile); do
        stri=$i
        if [[ $i -lt 10 ]]; then
            stri="0$i"
        fi
        mkdir $stri
        ln $name $stri
        cp jobDiffeomorphicMR.sh $stri
        i=$[$i +1]
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
############################Unknown##################################
echo Unknown option \'$1\'. The available options are "(c)"lean, "(s)"plit, s"(u)"bmit.
exit 0

