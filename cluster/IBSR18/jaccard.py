from ibsrutils import *
import nibabel as nib

def decompose_path(fname):
    dirname=os.path.dirname(fname)+'/'
    base=os.path.basename(fname)
    no_ext = os.path.splitext(base)[0]
    while(no_ext !=base):
        base=no_ext
        no_ext =os.path.splitext(base)[0]
    ext = os.path.basename(fname)[len(no_ext):]
    return dirname, base, ext


with open("jaccard_pairs.lst") as input:
    names = [s.split() for s in input.readlines()]
    for r in names:
        moving_dir, moving_base, moving_ext = decompose_path(r[0])
        fixed_dir, fixed_base, fixed_ext = decompose_path(r[1])
        warped_name = "warpedDiff_"+moving_base+"_"+fixed_base+".nii.gz"
        computeJacard(r[2], warped_name)

