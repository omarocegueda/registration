import os
import numpy as np
from dipy.fixes import argparse as arg
parser = arg.ArgumentParser(
    description=
        "Transforms a centered affine transform text file (from ANTS) to a simple (uncentered) affine transform text file with the same format."
    )

parser.add_argument(
    'iname', action = 'store', metavar = 'iname',
    help = '''Input text file containing a centered affine transform in itk (ANTS) format.''')

parser.add_argument(
    'oname', action = 'store', metavar = 'oname',
    help = '''Output file name to store the un-centered affine transform in itk (ANTS) format.''', 
    default=None)



def split_fname(fname):
    directory=os.path.dirname(fname)
    if directory:
        directory+='/'
    
    base=os.path.basename(fname)
    noExt=os.path.splitext(base)[0]
    while(noExt!=base):
        base=noExt
        noExt=os.path.splitext(base)[0]

    base=os.path.basename(fname)
    extensions = base[len(noExt):]

    return directory, noExt, extensions

def readAntsAffine(fname):
    '''
    readAntsAffine('IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt')
    '''
    try:
        with open(fname) as f:
            lines=[line.strip() for line in f.readlines()]
    except IOError:
        print 'Can not open file: ', fname
        return
    if not (lines[0]=="#Insight Transform File V1.0"):
        print 'Unknown file format'
        return
    if lines[1]!="#Transform 0":
        print 'Unknown transformation type'
        return
    A=np.zeros((3,3))
    b=np.zeros((3,))
    c=np.zeros((3,))
    for line in lines[2:]:
        data=line.split()
        if data[0]=='Transform:':
            if data[1]!='MatrixOffsetTransformBase_double_3_3':
                print 'Unknown transformation type'
                return
        elif data[0]=='Parameters:':
            parameters=np.array([float(s) for s in data[1:]], dtype=np.float64)
            A=parameters[:9].reshape((3,3))
            b=parameters[9:]
        elif data[0]=='FixedParameters:':
            c=np.array([float(s) for s in data[1:]], dtype=np.float64)
    T=np.ndarray(shape=(4,4), dtype=np.float64)
    T[:3,:3]=A[...]
    T[3,:]=0
    T[3,3]=1
    T[:3,3]=b+c-A.dot(c)
    ############This conversion is necessary for compatibility between itk and nibabel#########
    conversion=np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    T=conversion.dot(T.dot(conversion))
    ###########################################################################################
    return T


def writeAntsAffine(T, fname):
    A = np.zeros((3,3))
    b = np.zeros(3)
    A[:3, :3] = T[:3, :3]
    b = T[:3, 3]
    c = np.zeros(3)

    with open(fname, "w") as f:
        for s in ["#Insight Transform File V1.0", "#Transform 0", "Transform: MatrixOffsetTransformBase_double_3_3"]:
            f.write("%s\n"%s)
        f.write("Parameters: ")
        for x in A.reshape(-1):
            f.write("%0.18f "%x)
        for x in b:
            f.write("%0.18f "%x)
        f.write("\nFixedParameters: ")
        for x in c:
            f.write("%0.16f "%x)
        f.write("\n")

def uncenter_matrix_file(iname, oname=None):
    T = readAntsAffine(iname)

    #re-convert to itk
    conversion=np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
    T=conversion.dot(T.dot(conversion))
    #

    if(oname is None):
        directory, base, extensions = split_fname(iname)
        oname = directory+"u_"+base+extensions

    writeAntsAffine(T, oname)


if __name__ == "__main__":
    r"""
    python .\uncenter.py .\results\ANTSAffineIBSRToIBSR\test.txt .\results\ANTSAffineIBSRToIBSR\u_test.txt
    """
    params = parser.parse_args()
    uncenter_matrix_file(params.iname, params.oname)
