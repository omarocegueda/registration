#######################################################################
# Author: Omar Ocegueda (omar@cimat.mx)
#######################################################################
import sys
import os
import fnmatch
import shutil
import subprocess
import errno
registrationScriptName='/home/omar/code/registration/dipyreg.py'
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes":True,   "y":True,  "ye":True,
             "no":False,     "n":False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

if __name__=='__main__':
    argc=len(sys.argv)
    #############################No parameters#############################
    if not sys.argv[1]:
        print 'Please specify an action: c "(clean)", s "(split)", u "(submit)", o "(collect)"'
        sys.exit(0)
    #############################Clean#####################################
    if sys.argv[1]=='c':
        if not os.path.isdir('results'):
            cleanAnyway=query_yes_no("It seems like you have not collected the results yet. Clean anyway? (y/n)")
            if not cleanAnyway:
                sys.exit(0)
        dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
        for name in dirNames:
            shutil.rmtree(name)
        sys.exit(0)
    #############################Split####################################
    if sys.argv[1]=='s':
        if argc<3:
            print 'Please specify a text file containing the names of the files to register'
            sys.exit(0)
        try:
            with open(sys.argv[2]) as f:
                lines=f.readlines()
        except IOError:
            print 'Could not open file:', sys.argv[2]
            sys.exit(0)
        names=[line.strip().split() for line in lines]
        nlines=len(names)
        for i in range(nlines):
            if not names[i]:
                continue
            print 'Splitting reference:',names[i][0]
            reference=names[i]
            stri='0'+str(i+1) if i+1<10 else str(i+1)
            for j in range(nlines):
                if i==j:
                    continue
                if not names[j]:
                    continue
                target=names[j]
                strj='0'+str(j+1) if j+1<10 else str(j+1)
                dirName=strj+'_'+stri
                mkdir_p(os.path.join(dirName,'target'))
                mkdir_p(os.path.join(dirName,'reference'))
                mkdir_p(os.path.join(dirName,'warp'))
                subprocess.call('ln '+target[0]+' '+dirName+'/target', shell=True)
                subprocess.call('ln '+reference[0]+' '+dirName+'/reference', shell=True)
                subprocess.call('ln jobFullSyNEM.sh '+dirName, shell=True)
                subprocess.call('ln '+registrationScriptName+' '+dirName, shell=True)
                for w in target[1:]:
                    subprocess.call('ln '+w+' '+dirName+'/warp', shell=True)
        sys.exit(0)
    ############################Submit###################################
    if sys.argv[1]=='u':
        dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
        for name in dirNames:
            os.chdir('./'+name)
            subprocess.call('qsub jobFullSyNEM.sh -d .', shell=True)
            os.chdir('./..')
        sys.exit(0)
    ############################Collect##################################
    if sys.argv[1]=='o':
        mkdir_p('results')
        dirNames=[name for name in os.listdir(".") if os.path.isdir(name) and fnmatch.fnmatch(name, '[0-9]*')]
        for name in dirNames:
            subprocess.call('mv '+os.path.join(name,'*.nii.gz')+' results', shell=True)
            subprocess.call('mv '+os.path.join(name,'*.txt')+' results', shell=True)
            subprocess.call('mv '+os.path.join(name,'*.e*')+' results', shell=True)
            subprocess.call('mv '+os.path.join(name,'*.o*')+' results', shell=True)
        sys.exit(0)
    ############################Unknown##################################
    print 'Unknown option "'+sys.argv[1]+'". The available options are "(c)"lean, "(s)"plit, s"(u)"bmit, c"(o)"llect.'
