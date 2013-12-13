#!/bin/bash
#######################################################################
# Author: Omar Ocegueda (omar@cimat.mx)
# This script looks into all folders whose name start with a digit and
# checks the error log file from the cluster (*.e*). If the file is
# empty, it means that there were no execution errors. Otherwise,
# it erases the output and error log files and resubmits the job
# (whatever file whose name has 'job' as prefix)
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

