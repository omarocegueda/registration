#!/bin/bash
#######################################################################
# Author: Omar Ocegueda (omar@cimat.mx)
# This script creates a directory containing the necessary data 
# (a link to the data) to affine-register two IBSR brains 
# per node in the cluster. It receives as input a text file containing
# the absolute path of all the desired IBSR files to be registered.
# The first listed brain is considered the reference.
# If 'c' is given as parameter, it 'cleans' the working directory
# by force-removing all the subdirectories.
#
# Usage: ./splitANTSAffine.sh [option][fname]
# Example (clean): ./splitANTSAffine.sh c
# Example (split, all to first): ./splitANTSAffine.sh s names.txt
# Example (split, first to all): ./splitANTSAffine.sh s names.txt i
# Example (submit): ./splitANTSAffine.sh u
# Example (collect):./splitANTSAffine.sh o
#
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
    namesFile=$2
    if [ -z "$namesFile" ]; then
        echo Please specify the name of a text file containing the names of the IBSR input files to split
        exit 0
    fi
    if [ "$3" == "i" ]; then
        echo Preparing to register unique target: map first to the rest
    else
        echo Preparing to register unique reference: map all to first
    fi
    i=-1
    first=""
    for other in $(cat $namesFile); do
        i=$[$i +1]
        if [[ $i -eq 0 ]]; then
            first=$other
            continue
        fi
        stri="$i"
        if [[ $i -lt 10 ]]; then
            stri="0$i"
        fi
        mkdir $stri
        mkdir $stri/reference
        mkdir $stri/target
        if [ "$3" == "i" ]; then
            ln $first $stri/target
            ln $other $stri/reference
        else
            ln $other $stri/target
            ln $first $stri/reference
        fi
        cp jobANTSAffine.sh $stri
    done
    exit 0
fi
############################Submit###################################
if [ "$1" == "u" ]; then
    for name in $(ls -d */); do
        echo Submitting: \'$name\'
        cd $name
        qsub jobANTSAffine.sh -d .
        cd ..
    done
    exit 0
fi
############################Collect##################################
if [ "$1" == "o" ]; then
    mkdir -p results
    for dir in $(ls -d [0-9]*/); do
        cp $dir/*.gz results
        cp $dir/*.txt results
    done
    
fi
############################Unknown##################################
echo Unknown option \'$1\'. The available options are "(c)"lean, "(s)"plit, s"(u)"bmit, c"(o)"llect.
exit 0

