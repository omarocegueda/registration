##########################################################
# Author: Omar Ocegueda (omar@cimat.mx)
# This script generates a list of specific IBSR file names
# Usage: listIBSRNames.sh [suffix]
# Example: ./listIBSRNames.sh '_seg_ana'
#
# Note: Please edit the DATA_DIR variable to the location
# of your IBSR data
################Configuration#############################
DATA_DIR=/home/omar/data/IBSR_nifti_stripped
##########################################################
initDir=$(pwd)
suffix="$1"
if [ -z "$suffix" ]; then
    echo "Please specify a suffix (e.g. '_seg_ana') to select specific file types."
    exit 0
fi
cd $DATA_DIR
for name in $(ls -d */ | cut -c 1-7); do
    echo $DATA_DIR/$name/${name}${suffix}.nii.gz
done
cd $initialDir
##########################################################

